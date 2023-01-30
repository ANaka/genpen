import functools
import itertools
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import List

import bezier
import numpy as np
import shapely.affinity as sa
import shapely.geometry as sg
import shapely.ops as so
import vsketch
from scipy import stats as ss
from shapely import speedups
from shapely.errors import TopologicalError
from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                              MultiPolygon, Point, Polygon, box)
from tqdm import tqdm

SVG_SAVEDIR = '/home/naka/art/plotter_svgs'

def get_left(geom):
    return geom.bounds[0]

def get_bottom(geom):
    return geom.bounds[1]

def get_right(geom):
    return geom.bounds[2]

def get_top(geom):
    return geom.bounds[3]

def get_width(geom):
    return get_right(geom) - get_left(geom)

def get_height(geom):
    return get_top(geom) - get_bottom(geom)

### parameters 
class DataClassBase(object):
    def asdict(self):
        return asdict(self)

# @dataclass
# class ParamSeq(DataClassBase):


class Shape(object):
    def __init__(self, p:sg.Polygon):
        self._p = p
        self.fill = None
        
    def _repr_svg_(self):
        return self.p._repr_svg_()
    
    @property
    def p(self):
        return self._p
    
    @p.setter
    def p(self, p):
        self._p = p
    
    @property
    def width(self):
        return get_width(self._p)
    
    @property
    def height(self):
        return get_height(self._p)
    
    @property
    def left(self):
        return get_left(self._p)
    
    @property
    def right(self):
        return get_right(self._p)
    
    @property
    def top(self):
        return get_top(self._p)
    
    @property
    def bottom(self):
        return get_bottom(self._p)
    
    @property
    def boundary(self):
        return self._p.boundary

class Poly(Shape):
    def __init__(self, p:sg.Polygon):
        self.p = p
        self.fill = None
        
    def _repr_svg_(self):
        return self.p._repr_svg_()
    
    def scale_tran(self, d_buffer, d_translate, angle, cap_style=2, join_style=2):
        return scale_tran(self.p)

    def hatch(self, angle, spacing):
        return hatchbox(self.p, angle, spacing)
    
    def fill_scale_trans(self, d_buffers, d_translates, angles, cap_style=2, join_style=2):
        ssps = scale_trans(self.p, d_buffers, d_translates, angles, cap_style=cap_style, join_style=join_style)
        self.fill = merge_LineStrings([p.boundary for p in ssps])
        
    def fill_hatch(self, angle, spacing):
        self.fill = hatchbox(self.p, angle, spacing)
    
    @property
    def intersection_fill(self):
        try:
            ifill = self.fill.intersection(self.p)
        except TopologicalError:
            self.p = self.p.buffer(1e-6)
            ifill = self.fill.intersection(self.p)
        try:
            return collection_to_mls(ifill)
        except:
            return MultiLineString([ifill])
        
    def get_random_point(self):
        return get_random_point_in_polygon(self.p)


def centered_box(point, width, height):
    return sg.box(point.x-width/2, point.y-height/2, point.x+width/2, point.y+height/2)

def overlay_grid(poly, xstep, ystep, flatmesh=False):
    '''(xbins, ybins), (xs, ys) = gp.overlay_grid(poly=drawbox, xstep=5, ystep=5, flatmesh=True)'''
    xmin, ymin, xmax, ymax = poly.envelope.bounds
    xbins=np.arange(xmin, xmax, xstep)
    ybins=np.arange(ymin, ymax, ystep)
    bins = (xbins, ybins)
    grid = np.meshgrid(xbins, ybins)
    if flatmesh:
        return bins, (grid[0].ravel(), grid[1].ravel())
    else:
        return bins, grid

def get_random_points_in_polygon(polygon, n_points=1, xgen=None, ygen=None):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    if xgen == None:
        xgen = lambda size=None: np.random.uniform(minx, maxx, size)
    if ygen == None:
        ygen = lambda size=None: np.random.uniform(miny, maxy, size)

    n_attempts = 0
    while True:
        point = Point((xgen(), ygen()))
        if polygon.contains(point):
            points.append(point)
        n_attempts += 1
        if n_attempts > (n_points * 20):
            print('too many attempts being rejected in get_random_points_in_polygon')
            break
        if len(points) == n_points:
            return points
        

def get_random_point_in_polygon(polygon):
    return get_random_points_in_polygon(polygon, n_points=1)[0]

def get_rad(circle, use_circumference=False):

    if use_circumference:
        return circle.boundary.length / (np.pi * 2)
    else:
        return (circle.bounds[2] - circle.bounds[0]) / 2
    
    
    
### Shading ###

def scale_tran(p, d_buffer, d_translate, angle, cap_style=2, join_style=2, **kwargs):
    xoff = np.cos(angle) * d_translate
    yoff = np.sin(angle) * d_translate
    bp = p.buffer(d_buffer, cap_style=cap_style, join_style=join_style, **kwargs)
    btp = sa.translate(bp, xoff=xoff, yoff=yoff)
    return btp


def scale_trans(p, d_buffers, d_translates, angles, cap_style=2, join_style=2, return_original=True, **kwargs):
    
    d_translates = ensure_collection(d_translates, length=len(d_buffers))
    angles = ensure_collection(angles, length=len(d_buffers))
    
    ssps = []
    if return_original:
        ssps.append(p)
    
    ssp = p
    for d_buffer, d_translate, angle in zip(d_buffers, d_translates, angles):
        ssp = scale_tran(ssp, d_buffer, d_translate, angle, cap_style, join_style, **kwargs)
        if ssp.area < np.finfo(float).eps:
            break
        ssps.append(ssp)
    return ssps

@dataclass
class ScaleTransPrms(DataClassBase):
    '''
    pt = Point(0,0)
    poly = Poly(pt.buffer(10))
    prms = ScaleTransPrms(
        n_iters=200,
        d_buffer=-0.25,
        d_translate_factor=0.7,
        angles=0,
    )
    poly.fill_scale_trans(**prms.prms)
    '''
    n_iters: int = 100
    d_buffer: float = -0.25
    d_translate_factor: float = 0.9
    d_translate: float = None
    angles: float = 0. # radians
    d_translates: list = field(default=None, init=False)
    def __post_init__(self):
        self.d_buffers = np.array([self.d_buffer] * self.n_iters)
        
        if self.d_translates == None:
            if self.d_translate != None:
                self.d_translates =  np.array([self.d_translate] * self.n_iters)
            else:
                self.d_translates = self.d_buffers * self.d_translate_factor
    
    @property
    def prms(self):
        varnames = ['d_buffers', 'd_translates', 'angles']
        return {var: getattr(self, var) for var in varnames}
    


def hatchbox(rect, angle, spacing):
    """
    returns a Shapely geometry (MULTILINESTRING, or more rarely,
    GEOMETRYCOLLECTION) for a simple hatched rectangle.

    args:
    rect - a Shapely geometry for the outer boundary of the hatch
           Likely most useful if it really is a rectangle

    angle - angle of hatch lines, conventional anticlockwise -ve

    spacing - spacing between hatch lines

    GEOMETRYCOLLECTION case occurs when a hatch line intersects with
    the corner of the clipping rectangle, which produces a point
    along with the usual lines.
    """

    (llx, lly, urx, ury) = rect.bounds
    centre_x = (urx + llx) / 2
    centre_y = (ury + lly) / 2
    diagonal_length = ((urx - llx) ** 2 + (ury - lly) ** 2) ** 0.5
    number_of_lines = 2 + int(diagonal_length / spacing)
    hatch_length = spacing * (number_of_lines - 1)

    # build a square (of side hatch_length) horizontal lines
    # centred on centroid of the bounding box, 'spacing' units apart
    coords = []
    for i in range(number_of_lines):
        if i % 2:
            coords.extend([((centre_x - hatch_length / 2, centre_y
                          - hatch_length / 2 + i * spacing), (centre_x
                          + hatch_length / 2, centre_y - hatch_length
                          / 2 + i * spacing))])
        else:
            coords.extend([((centre_x + hatch_length / 2, centre_y
                          - hatch_length / 2 + i * spacing), (centre_x
                          - hatch_length / 2, centre_y - hatch_length
                          / 2 + i * spacing))])
            
    # turn array into Shapely object
    lines = sg.MultiLineString(coords)
    # Rotate by angle around box centre
    lines = sa.rotate(lines, angle, origin='centroid', use_radians=False)
    # return clipped array
    return rect.intersection(lines)

def connect_hatchlines(hatchlines, dist_thresh):
    linestrings = list(hatchlines)
    merged_linestrings = []

    current_ls = linestrings.pop(0)
    while len(linestrings) > 0:
        next_ls = linestrings.pop(0)
        dist = Point(current_ls.coords[-1]).distance(Point(next_ls.coords[0]))
        if dist <= dist_thresh:
            current_ls = LineString(list(current_ls.coords) + list(next_ls.coords))
        else:
            merged_linestrings.append(current_ls)
            current_ls = next_ls
    merged_linestrings.append(current_ls)
    return merged_linestrings


def connected_hatchbox(rect, angle, spacing, dist_thresh):
    hatches = hatchbox(rect, angle, spacing)
    return hatches




def morsify(ls, buffer_factor=0.01):
    return ls.buffer(buffer_factor).buffer(-buffer_factor).boundary


def add_jittered_midpoints(ls, n_midpoints, xstd, ystd, xbias=0, ybias=0):
    eval_range = np.linspace(0., 1., n_midpoints+2)
    pts = np.stack([ls.interpolate(t, normalized=True) for t in eval_range])
    x_jitter = np.random.randn(n_midpoints, 1) * xstd + xbias
    y_jitter = np.random.randn(n_midpoints, 1) * ystd + ybias
    pts[1:-1] += np.concatenate([x_jitter, y_jitter], axis=1)
    return sg.asLineString(pts)


def LineString_to_jittered_bezier(ls, n_midpoints=1, xbias=0., xstd=0., ybias=0., ystd=0., normalized=True, n_eval_points=50):
    if normalized==True:
        xbias = xbias * ls.length
        xstd = xstd * ls.length
        ybias = ybias * ls.length
        ystd = ystd*ls.length

    jitter_ls = add_jittered_midpoints(ls, n_midpoints=n_midpoints, xbias=xbias, xstd=xstd, ybias=ybias, ystd=ystd,)
    curve1 = bezier.Curve(np.asfortranarray(jitter_ls).T, degree=n_midpoints+1)
    bez = curve1.evaluate_multi(np.linspace(0., 1., n_eval_points))
    return sg.asLineString(bez.T)


def circle_pack_within_poly(poly, rads, max_additions=np.inf, progress_bar=True):
    # max additions limit doesn't really work

    radi = iter(rads)
    if progress_bar:
        pbar = tqdm(total=len(rads))

    # init
    n_additions = 0
    rad = next(radi)
    circles = []
    all_circles = MultiPolygon()
    init_attempts = 0
    while len(circles) == 0:
        if init_attempts > 10:
            break
        try:
            pt = get_random_points_in_polygon(poly.buffer(-rad))[0]
            c = pt.buffer(rad)
            c.rad = rad
            if (not c.intersects(all_circles)) and (poly.contains(c)):
                circles.append(c)
                all_circles = so.unary_union([all_circles, c])
                n_additions += 1
        except:
            init_attempts += 1


    # main loop

    while True:
        try:

            while n_additions <= max_additions:
                at_least_one_addition = False
                circle_order = np.random.permutation(len(circles))
                for i in circle_order:
                    seed_circle = circles[i]
                    seed_rad = seed_circle.rad
                    search_ring = seed_circle.buffer(rad).boundary
                    search_locs = np.arange(0, search_ring.length, (2*rad))
                    scrambled_search_locs = np.random.permutation(search_locs)
                    for sl in scrambled_search_locs:
                        pt = search_ring.interpolate(sl)
                        c = pt.buffer(rad)
                        c.rad = rad
                        if (not c.intersects(all_circles)) and (poly.contains(c)):
                            circles.append(c)
                            all_circles = so.unary_union([all_circles, c])
                            at_least_one_addition = True
                            n_additions += 1
                if not at_least_one_addition:  # if not working, break and reduce rad
                    break
            rad = next(radi)
            n_additions = 0
            if progress_bar:
                pbar.update()
        except StopIteration:
            return all_circles

@dataclass
class RecursiveCirclePacker(object):
    poly_to_fill: Polygon
    fill_by_relative_radius: bool=True
    rad_seq_start: float =0.5
    rad_seq_end: float=0.1
    n_rads: float =30
    min_allowed_rad: float=0.2
    store_init_poly: bool=False
    n_fail_max: int = 20
    
    
    '''rcp = RecursiveCirclePacker(Point(0,0).buffer(10), 
    rad_seq_start=0.48, rad_seq_end=0.05, min_allowed_rad=2, n_rads=30,)
    rcp.run(1)
    rcp.unfilled_circles'''
    
    
    def __post_init__(self):
        self.all_circles = []
        if self.store_init_poly:
            self.all_circles.append(self.poly_to_fill)
    
    @property
    def fill_poly_rad(self):
        return get_rad(self.poly_to_fill)
    
    def gen_relative_rad_seq(self):
        return np.linspace(
            self.fill_poly_rad * self.rad_seq_start,
            self.fill_poly_rad * self.rad_seq_end,
            self.n_rads)
    
    def gen_absolute_rad_seq(self):
        return np.linspace(
            self.rad_seq_start,
            self.rad_seq_end,
            self.n_rads)
      
    @property
    def rad_seq(self):
        if self.fill_by_relative_radius:
            return self.gen_relative_rad_seq()
        else:
            return self.gen_absolute_rad_seq()
        
    def pack_current_poly(self, progress_bar=False):
        circles = circle_pack_within_poly(self.poly_to_fill, self.rad_seq, progress_bar=progress_bar)
        self.all_circles += list(circles)
        self.poly_to_fill.filled = True
        
    @property
    def filled_ind(self):
        return np.array([getattr(c, 'filled', False) for c in self.all_circles])
    
    @property
    def areas(self):
        return np.array([c.area for c in self.all_circles])
    
    @property
    def choice_prob(self):
        filtered_areas = ~self.filled_ind * self.areas
        return filtered_areas / filtered_areas.sum()
    
    def choose_next_poly(self):
        self.poly_to_fill = np.random.choice(self.all_circles, p=self.choice_prob)
    
    @property
    def circles(self):
        return merge_Polygons(self.all_circles)
    
    @property
    def boundary(self):
        return merge_LineStrings([p.boundary for p in self.all_circles])
    
    @property
    def unfilled_circles(self):
        return merge_Polygons([self.all_circles[i] for i in np.flatnonzero(~self.filled_ind)])
    
    @property
    def unfilled_boundary(self):
        return merge_LineStrings([self.all_circles[i].boundary for i in np.flatnonzero(~self.filled_ind)])
    
    def run(self, iter_max=1, progress_bar=False):
        n_fails = 0
        n_iters = 0
        if progress_bar:
            pbar = tqdm(total=iter_max)
            
        while (n_fails < self.n_fail_max) and (n_iters < iter_max):
            try:
                n_iters += 1
                self.pack_current_poly()
                self.choose_next_poly()
                n_fails = 0
                pbar.update()
            except:
                n_fails += 1
                self.choose_next_poly()


@dataclass
class NuzzleParams(DataClassBase):
    dilate_multiplier_min: float = 0.01
    dilate_multiplier_max: float = 0.1
    erode_multiplier:float = -1.1
    n_iters:int = 1
    dilate_join_style:int = 1
    dilate_cap_style:int = 1
    erode_join_style:int = 1
    erode_cap_style:int = 1


def get_rad(circle, use_circumference=False):

    if use_circumference:
        return circle.boundary.length / (np.pi * 2)
    else:
        return (circle.bounds[2] - circle.bounds[0]) / 2


def nuzzle_poly(poly, neighbors,
                dilate_multiplier_min,
                dilate_multiplier_max,
                erode_multiplier=-1.1,
                n_iters=1,
                dilate_join_style=1,
                dilate_cap_style=1,
                erode_join_style=1,
                erode_cap_style=1,
               ):
    rad = get_rad(poly)
    neighbor_union = so.unary_union(neighbors)
    for i in range(n_iters):
        d = rad * np.random.uniform(dilate_multiplier_min, dilate_multiplier_max)
        bc = poly.buffer(d, join_style=dilate_join_style, cap_style=dilate_cap_style)
        if bc.intersects(neighbor_union):
            e = d * erode_multiplier
            bc = bc.difference(neighbor_union).buffer(e,join_style=erode_join_style, cap_style=erode_cap_style)
        poly = bc
    return poly


def nuzzle_em(polys, nuzzle_params, n_iters=20):
    for n in range(n_iters):
        polys = list(polys)
        order = np.random.permutation(len(polys))
        for i in order:
            try:
                poly = polys[i]
                other_polys = MultiPolygon([c for j, c in enumerate(polys) if j!=i])
                new_poly = nuzzle_poly(poly, other_polys, **nuzzle_params.asdict())
                polys[i] = new_poly
            except:
                pass
        polys = merge_Polygons(polys)
    return merge_Polygons(polys)


def random_split(geoms, n_layers):
    splits = np.random.choice(n_layers, size=len(geoms))
    layers = []

    for i in range(n_layers):
        layers.append([geoms[j] for j in np.nonzero(splits==i)[0]])
    return layers



def circle_growth(poly, rads, obj_func, max_additions=np.inf, progress_bar=True):
    # max additions limit doesn't really work

    radi = iter(rads)
    if progress_bar:
        pbar = tqdm(total=len(rads))

    # init
    n_additions = 0
    rad = next(radi)
    circles = []
    all_circles = MultiPolygon()
    while len(circles) == 0:
        pt = get_random_points_in_polygon(poly.buffer(-rad))[0]
        c = pt.buffer(rad)
        c.rad = rad
        if (not c.intersects(all_circles)) and (poly.contains(c)):
            circles.append(c)
            all_circles = so.unary_union([all_circles, c])
            n_additions += 1

    # main loop

    while True:
        try:

            while n_additions <= max_additions:
                at_least_one_addition = False
                circle_order = np.random.permutation(len(circles))
                for i in circle_order:
                    seed_circle = circles[i]
                    seed_rad = seed_circle.rad
                    search_ring = seed_circle.buffer(rad).boundary
                    search_locs = np.arange(0, search_ring.length, (2*rad))
                    scrambled_search_locs = np.random.permutation(search_locs)
                    for sl in scrambled_search_locs:
                        pt = search_ring.interpolate(sl)
                        c = pt.buffer(rad)
                        c.rad = rad
                        if (not c.intersects(all_circles)) and (poly.contains(c)):
                            circles.append(c)
                            all_circles = so.unary_union([all_circles, c])
                            at_least_one_addition = True
                            n_additions += 1
                if not at_least_one_addition:  # if not working, break and reduce rad
                    break
            rad = next(radi)
            n_additions = 0
            if progress_bar:
                pbar.update()
        except StopIteration:
            return all_circles



# Cell





def buffer_individually(geoms, distance, cap_style=2, join_style=2):
    n_geoms = len(geoms)
    ds = ensure_collection(distance, n_geoms)
    css = ensure_collection(cap_style, n_geoms)
    jss = ensure_collection(join_style, n_geoms)
    bgs = []
    for i in range(n_geoms):
        bg = geoms[i].buffer(ds[i], cap_style=css[i], join_style=jss[i])
        bgs.append(bg)
    return MultiPolygon(bgs)


def merge_LineStrings(mls_list):
    merged_mls = []
    for mls in mls_list:
        if getattr(mls, 'type') == 'MultiLineString':
            merged_mls += list(mls)
        elif getattr(mls, 'type') == 'LineString':
            merged_mls.append(mls)
    return sg.MultiLineString(merged_mls)

def merge_Polygons(mp_list):
    merged_mps = []
    for mp in mp_list:
        if type(mp)==list:
            merged_mps += list(mp)
        elif getattr(mp, 'type') == 'MultiPolygon':
            merged_mps += list(mp)
        elif getattr(mp, 'type') == 'Polygon':
            merged_mps.append(mp)
    return sg.MultiPolygon(merged_mps)


def collection_to_mls(collection):
    lss = [g for g in collection if 'LineString' in g.type]
    lss = [ls for ls in lss if ls.length > np.finfo(float).eps]
    return merge_LineStrings(lss)


def occlude(top, bottom, distance=1e-6):
    try:
        return bottom.difference(top)
    except TopologicalError:
        return bottom.buffer(distance).difference(top.buffer(distance))
    
    
# class ParticleCluster(object):
    
#     def __init__(
#         self,
#         pos,
#         perlin_grid,
#     ):
#         self.pos = Point(pos)
#         self.pg = perlin_grid
#         self.particles = []
        
#     def gen_start_pts_gaussian(
#             self,
#             n_particles=10,
#             xloc=0.,
#             xscale=1.,
#             yloc=0.,
#             yscale=1.,
#         ):
#         xs = self.pos.x + ss.norm(loc=xloc, scale=xscale).rvs(n_particles)
#         ys = self.pos.y + ss.norm(loc=yloc, scale=yscale).rvs(n_particles)
#         self.start_pts = [Point((x,y)) for x,y in zip(xs, ys)]
        
#     def init_particles(self, start_bounds=None):
#         for pt in self.start_pts:
#             p = Particle(pos=pt, grid=self.pg)
#             if start_bounds == None:
#                 self.particles.append(p)
#             elif start_bounds.contains(p.pos):
#                 self.particles.append(p)
                
#     @functools.singledispatchmethod           
#     def step(self, n_steps):
#         for p,n in zip(self.particles, n_steps):
#             for i in range(n):
#                 p.step()
    
#     @step.register
#     def _(self, n_steps: int):
#         n_steps = [n_steps] * len(self.particles)
#         for p,n in zip(self.particles, n_steps):
#                 for i in range(n):
#                     p.step()
                    
#     @property
#     def lines(self):
#         return MultiLineString([p.line for p in self.particles])
    
    
    

def consolidate_polygons(polys):
    '''For if polygons are inside each other '''
    consolidated_polys = []
    for p0 in polys:
        other_polys = [p for p in polys if p != p0]
        p0_contains = []
        p0_is_within = []
        for p1 in other_polys:
            if p0.contains(p1):
                p0_contains.append(p1)
            elif p0.within(p1):
                p0_is_within.append(p1)
        
        new_p = Polygon(p0)
        if not any(p0_is_within):
            for p in p0_contains:
                new_p = robust_difference(p0, p)
            consolidated_polys.append(new_p)
    return consolidated_polys

def robust_difference(p0, p1, buffer_distance=1e-6):
    try:
        return p0.difference(p1)
    except TopologicalError:
        return p0.buffer(buffer_distance).difference(p1.buffer(buffer_distance))
    
    
def center_at(geom, target_point, return_transform=False, use_centroid=False):
    transform = {}
    if use_centroid:
        init_point = geom.centroid
    else:
        init_point = box(*geom.bounds).centroid
    transform['xoff'] = target_point.x - init_point.x
    transform['yoff'] = target_point.y - init_point.y
    
    translated = sa.translate(geom, **transform)
    if return_transform:
        return translated, transform
    else:
        return translated

def scale_like(geom, target_geom, preserve_aspect_ratio=True, use_smaller=True, return_transform=False):
    width_g = get_width(geom)
    height_g = get_height(geom)

    width_tg = get_width(target_geom)
    height_tg = get_height(target_geom)

    xfact = width_tg / width_g
    yfact = height_tg / height_g
    if preserve_aspect_ratio and use_smaller:
        fact = min([xfact, yfact])
        transform = {'xfact': fact, 'yfact':fact}
        scaled = sa.scale(geom, **transform)
    elif preserve_aspect_ratio and not use_smaller:
        fact = max([xfact, yfact])
        transform = {'xfact': fact, 'yfact':fact}
        scaled = sa.scale(geom, **transform)
    else:
        transform = {'xfact': xfact, 'yfact':yfact}
        scaled = sa.scale(geom, **transform)
        
    if return_transform:
        return scaled, transform
    else:
        return scaled
    
    
def make_like(p, target, return_transform=False):
    'rescale and center, good for making it fit in a drawbox'
    scaled, transform = scale_like(p, target, return_transform=True)
    transformed_poly, translate_transform = center_at(scaled, target.centroid, return_transform=True)
    
    transform.update(translate_transform)
    if return_transform:
        return transformed_poly, transform
    else:
        return transformed_poly
    
    
def gaussian_random_walk(n, step_init=1, step_mu=0., step_std=1, scale=True):
    ys = []
    y = step_init
    if scale:
        step_mu /= n
        step_std /= n
    for i in range(n):
        ys.append(y)
        y += np.random.randn() * step_std + step_mu
    return np.array(ys)


@functools.singledispatch
def make_callable(arg):
    pass

@make_callable.register(int)
@make_callable.register(float)
@make_callable.register(bool)
def _(arg):
    class CallableNumeric(type(arg)):
        
        def __call__(self, *args, **kwargs):
            return self.real
        
    return CallableNumeric(arg)

@make_callable.register(list)
@make_callable.register(tuple)
@make_callable.register(np.ndarray)
def _(arg):
    class CallableSequence(type(arg)):
        
        def __init__(self, arg):
            super().__init__(arg)
            self._generator = iter(self)
            
        def __call__(self, *args, **kwargs):
            return next(self._generator)
        
    return CallableSequence(arg)

@make_callable.register(Callable)
def _(arg):
    return arg

        
class DelauneyLattice(object):
    
    def __init__(
        self,
        vertices: MultiPoint,
        max_length_filter_gen,
        min_length_filter_gen,
        buffer_size_gen,
        cap_style=None,
        join_style=None,
        buffer_individually=True,
    ):
        
        self.vertices = vertices
        self.max_length_filter_gen = max_length_filter_gen
        self.min_length_filter_gen = min_length_filter_gen
        self.buffer_size_gen = buffer_size_gen
        self.buffer_individually = buffer_individually
        
        if cap_style is None:
            self._cs = make_callable(2)
        if join_style is None:
            self._js = make_callable(3)
            
        self.edges = MultiLineString(so.triangulate(vertices, edges=True))
    
    @staticmethod
    def _buffer(x, distance, **kwargs):
        return x.buffer(distance, **kwargs)
    
    @property
    def _buffer_func(self):
        return functools.partial(
            self._buffer,
            cap_style=self._cs,
            join_style=self._js)
    
    @property
    def filt_edges(self):
        filt_edges = []
        for ls in self.edges:
            too_long = ls.length > self.max_length_filter_gen(ls)
            too_short = ls.length < self.min_length_filter_gen(ls)
            if not too_long and not too_short:
                filt_edges.append(ls)
        return MultiLineString(filt_edges)
    
    @property
    def polys(self):
        if self.buffer_individually:
            ps = []
            for ls in self.filt_edges:
                d = self.buffer_size_gen(ls)
                p = self._buffer_func(ls, d)
                ps.append(p)
            return MultiPolygon(ps)
        if not self.buffer_individually:
            d = self.buffer_size_gen(self.filt_edges)
            return self._buffer_func(self.filt_edges, d)
        
        
@dataclass
class AffineMatrix(DataClassBase):
    xoff: float = 0.
    yoff: float = 0.
    xfact: float = 1.
    yfact: float = 1.
    angle: float = 0.  # in degrees
    
    @property
    def rotation_matrix(self):
        a = self.angle
        return np.array(
            [
                [np.cos(a), -np.sin(a), 0.],
                [np.sin(a), np.cos(a), 0.],
                [0, 0, 1]
            ]
        )
    
    @property
    def translation_matrix(self):
        return np.array(
            [
                [1, 0, -self.xoff],
                [0, 1, -self.yoff],
                [0, 0, 1]
            ]
        )
    
    @property
    def scaling_matrix(self):
        return np.array(
            [
                [self.xfact, 0, 1],
                [0, self.yfact, 1],
                [0, 0, 1]
            ]
        )
    
    @property
    def A(self):
        return self.scaling_matrix @ self.translation_matrix @ self.rotation_matrix
    
    @property
    def A_flat(self):
        A = self.A.ravel()
        return [A[0], A[1], A[3], A[4], A[2], A[5]]
        
    '''
    [a, b, d, e, xoff, yoff]
    [
    [a, b, xoff]
    [d, e, yoff]
    ]
    '''


def scalar_to_collection(scalar, length):
    stype = type(scalar)
    return (np.ones(length) * scalar).astype(stype)

def ensure_collection(x, length):
    if np.iterable(x):
        assert len(x) == length
        return x
    else:
        return scalar_to_collection(x, length)
    
def random_split(geoms, n_layers):
    splits = np.random.choice(n_layers, size=len(geoms))
    layers = []

    for i in range(n_layers):
        layers.append([geoms[j] for j in np.nonzero(splits==i)[0]])
    return layers


def polygonize_circle(circle, n_corners):
    angles = np.linspace(0, 1, n_corners+1)
    corners = [circle.boundary.interpolate(a, normalized=True) for a in angles[:-1]]
    return Polygon(corners)

def reg_polygon(point, radius, n_corners, **kwargs):
    circle = point.buffer(radius)
    return polygonize_circle(circle, n_corners)

@dataclass
class RegPolygon(object):
    point: Point
    radius: float=1.
    n_corners: int = 6
    rotation: float = 0.  #degrees
    buffer_kwargs: dict = field(default_factory=dict)
   
    @property
    def poly(self):
        poly = reg_polygon(self.point, self.radius, self.n_corners, **self.buffer_kwargs)
        return sa.rotate(poly, self.rotation, origin='centroid')
    
    @property
    def corners(self):
        return MultiPoint(self.poly.boundary.coords)[:-1]
@dataclass
class StellarSnowflake(object):
    point_f0: Point = Point((0,0))
    radius_f0: float=1.
    n_corners_f0: int = 6
    rotation_f0: float = 0.  #degrees
    radius_f1: float=0.3
    n_corners_f1: int = 6
    rotation_f1: float = 0.  #degrees    
    
    @property
    def f0(self):
        return RegPolygon(self.point_f0,
                          self.radius_f0, self.n_corners_f0, self.rotation_f0)
    
    @property
    def poly_center(self):
        return self.f0.poly
    
    @property
    def corner_polys(self):
        corner_polys = []
        for p in self.f0.corners:
            cp = RegPolygon(point=p, 
                             radius=self.radius_f1, 
                             n_corners=self.n_corners_f1,
                             rotation=self.rotation_f1,
                           ).poly
            corner_polys.append(cp) 
        return MultiPolygon(corner_polys)
    
    @property
    def multipolygon(self):
        return merge_Polygons([self.poly_center, self.corner_polys])
    
    @property
    def poly(self):
        return so.unary_union([self.poly_center, self.corner_polys])
    
def reverse_LineString(linestring):
    return LineString(list(linestring.coords)[::-1])


class Fill(Shape):
    
    def __init__(self, p):
        self._p = p
        
    def _repr_svg_(self):
        return self.fill._repr_svg_()
    
    @property
    def boundary(self):
        return self._p.boundary
    
    @property
    def fill(self):
        return self.fill_poly(self._p)
    
    @property
    def _all_geoms(self):
        return sg.GeometryCollection([self.boundary, self.fill])

class Angle(object):
    
    @property
    def deg(self):
        return self._deg
    
    @deg.setter
    def deg(self, deg):
        self._deg = deg
        self._rad = np.deg2rad(deg)
    
    @property
    def rad(self):
        return self._rad
    
    @rad.setter
    def rad(self, rad):
        self._deg = np.rad2deg(rad)
        self._rad = rad
        
    
    def __init__(self, deg=None, rad=None):
        if (deg is not None) and (rad is not None):
            print('WARNING: arguments entered for both deg and rad; defaulting to using deg')
        if deg is not None:
            self.deg = deg
        elif rad is not None:
            self.rad = rad
            
            
    def __repr__(self):
        return f'deg = {self.deg} \nrad = {self.rad:.3}'

@dataclass
class Filler(DataClassBase):
        
    def fill_poly(self, poly):
        return self.fill_func(poly)
    
    
    
@dataclass  
class HatchFill(Shape):
    
    def __init__(
        self,
        poly_to_fill:sg.Polygon,
        degrees:float=0.,
        spacing:float=0.5,  # usually mm
        alternate_direction:bool=True,
        fill_inscribe_buffer=1.1,
    ):
        self._ptf = poly_to_fill
        self.degrees = degrees
        self.spacing = spacing
        self.alternate_direction = alternate_direction
        self.fill_inscribe_buffer = fill_inscribe_buffer
    
    @property
    def angle(self):
        return Angle(deg=self.degrees)
    
    @property
    def inscribe_radius(self):
        d_from_furthest_vertex_to_centroid = self._ptf.hausdorff_distance(self._ptf.centroid)
        return d_from_furthest_vertex_to_centroid * self.fill_inscribe_buffer
    
    @property
    def inscribe_diameter(self):
        return self.inscribe_radius * 2
    
    @property
    def envelope_inscribe(self):
        return self._ptf.centroid.buffer(self.inscribe_radius)
    
    @property
    def envelope(self):
        # rotating geom around centroid will always be inside this
        return Shape(box(*self.envelope_inscribe.bounds))
    
    @property
    def _e(self):
        return self.envelope
    
    @property
    def spacings(self):
        try:
            len(self.spacing) > 1
            spacings = self.spacing
        except TypeError:
            # if spacing is a number, make list of uniform spacings
            spacings = np.arange(0, self.inscribe_diameter, self.spacing)
        return spacings
    
    @property
    def lines(self):
        left = self.envelope.left
        right = self.envelope.right
        top = self.envelope.top
        lines = []
        for ii, _spacing in enumerate(self.spacings):
            y = top - _spacing
            line = LineString(((left, y), (right, y)))
            lines.append(line)
            
        if self.alternate_direction:
            for i in range(len(lines)):
                if i % 2:
                    lines[i] = reverse_LineString(lines[i])
        
        return sg.MultiLineString(lines)
    
    @property
    def rotated_lines(self):
        return sa.rotate(self.lines, self.angle.deg, origin=self._ptf.centroid, use_radians=False)
    
    @property
    def fill(self):
        return self._ptf.intersection(self.rotated_lines)
    
    @property
    def _p(self):
        return self.fill
        
    
    

@dataclass  
class BezierHatchFill(HatchFill):
    
    def __init__(
        self,
        poly_to_fill:sg.Polygon,
        xjitter_func,
        yjitter_func,
        degrees:float=0.,
        spacing:float=0.5,  # usually mm
        alternate_direction:bool=True,
        fill_inscribe_buffer=1.01,
        n_nodes_per_line=20,
        n_eval_points=100,
        
    ):
        self._ptf = poly_to_fill
        self.degrees = degrees
        self.spacing = spacing
        self.alternate_direction = alternate_direction
        self.fill_inscribe_buffer = fill_inscribe_buffer
        self.n_nodes_per_line = n_nodes_per_line
        self.init_nodes = None
        self.xjitter_func = make_callable(xjitter_func)
        self.yjitter_func = make_callable(yjitter_func)
        self.n_eval_points = n_eval_points
    
    def initialize_nodes(self):
        # here to be expanded
        self.init_node_xs = np.linspace(self._e.left, self._e.right, self.n_nodes_per_line)
        self.init_node_ys = np.zeros_like(self.init_node_xs)
        self.init_nodes = np.stack([self.init_node_xs, self.init_node_ys])
    
    @property
    def node_sets(self):
        if self.init_nodes is None:
            self.initialize_nodes()
        xs = self.init_node_xs.copy()
        ys = self.init_node_ys.copy()
        nodes = np.stack([xs, ys])
        node_sets = [nodes.copy()]
        for ii in range(1, len(self.spacings)):
            xs += self.xjitter_func(xs.shape)
            ys += self.yjitter_func(ys.shape)
            nodes = np.stack([xs, ys])
            node_sets.append(nodes)
        return node_sets
        
    @property
    def ft_node_sets(self):
        return [np.asfortranarray(n) for n in self.node_sets]
    
    @property
    def curves(self):
        curves = []
        for node_set in self.ft_node_sets:
            curve = bezier.Curve(node_set, degree=(node_set.shape[1]-1))
            curves.append(curve)
        return curves
    
    @property
    def _interpolated_curves(self):
        i_curves = []
        for curve in self.curves:
            eval_points = np.linspace(0, 1, self.n_eval_points)
            x, y = curve.evaluate_multi(eval_points)
            i_curves.append(np.stack([x, y]).T)
        return i_curves
    
    @property
    def lines(self):
        lines = []
        
        for ii, curve in enumerate(self._interpolated_curves):
            spacing = self.spacings[ii]
            y = self._e.top - spacing
            _curve = curve.copy()
            _curve[:, 1] += y
            line = LineString(_curve)
            lines.append(line)
            
        if self.alternate_direction:
            for i in range(len(lines)):
                if i % 2:
                    lines[i] = reverse_LineString(lines[i])
        
        lines = [l for l in lines if l.length > 0]
        
        return sg.MultiLineString(lines)
    
def random_line_subdivide_gen(poly, x0gen=None, x1gen=None):
    x0 = x0gen()
    x1 = x1gen()
    return LineString([poly.boundary.interpolate(x, normalized=True) for x in [x0, x1]])
    
def random_line_subdivide(poly, x0=None, x1=None):
    if x0 is None:
        x0 = np.random.uniform(0,1)
    if x1 is None:
        x1 = (x0 + 0.5) % 1
    return LineString([poly.boundary.interpolate(x, normalized=True) for x in [x0, x1]])

def random_bezier_subdivide(poly, x0=None, x1=None, n_eval_points=50):
    if x0 is None:
        x0 = np.random.uniform(0.2, 0.4)
    if x1 is None:
        x1 = np.random.uniform(0.6, 0.8)
    line = np.asfortranarray(random_line_subdivide(poly, x0, x1))
    bez_array = np.stack([line[0], poly.centroid, line[1]]).T
    curve1 = bezier.Curve(bez_array, degree=2)
    bez = curve1.evaluate_multi(np.linspace(0., 1., n_eval_points))
    return sg.asLineString(bez.T)

def split_along_longest_side_of_min_rectangle(poly, xgen=None):
    if xgen is None:
        xgen = make_callable(0.5)
    mrrc = poly.minimum_rotated_rectangle.boundary.coords
    sides = [LineString([mrrc[i], mrrc[i+1]]) for i in range(4)]
    longest_sides = [sides[i] for i in np.argsort([-l.length for l in sides])[:2]]
    bps = [ls.interpolate(xgen(), normalized=True)for ls in longest_sides]

    return LineString([so.nearest_points(bp, poly.boundary)[1] for bp in bps])

def split_poly(poly, line):
    return list(poly.difference(line.buffer(1e-6)))

def recursive_split(poly, split_func=random_line_subdivide, p_continue=0.7, depth=0, depth_limit=15, buffer_kwargs=None):
    
    if buffer_kwargs is None:
        buffer_kwargs = {'distance':0}
    polys = list(poly.difference(split_func(poly).buffer(1e-6)))
    split_polys = []
    
    for i, p in enumerate(polys):
        continue_draw = np.random.binomial(n=1, p=p_continue)
        
        if continue_draw and (depth<depth_limit):
            
            split_polys += recursive_split(
                p, split_func=split_func, p_continue=p_continue, 
                depth=depth+1, depth_limit=depth_limit,
                buffer_kwargs=buffer_kwargs
            ) 
        else:
            split_polys.append(p.buffer(**buffer_kwargs))
    return split_polys

def recursive_split_frac_buffer(poly, split_func=random_line_subdivide, p_continue=0.7, depth=0, depth_limit=15, buffer_kwargs=None, buffer_frac=-0.1):
    try:
        if buffer_kwargs is None:
            buffer_kwargs = {'join_style':2, 'cap_style':2}
        polys = list(poly.difference(split_func(poly).buffer(1e-6)))
        split_polys = []

        for i, p in enumerate(polys):
            continue_draw = np.random.binomial(n=1, p=p_continue)
            distance=p.centroid.distance(p.boundary)*buffer_frac
            bp = p.buffer(distance=distance, **buffer_kwargs)
            if continue_draw and (depth<depth_limit):

                split_polys += recursive_split_frac_buffer(
                    bp, split_func=split_func, p_continue=p_continue, 
                    depth=depth+1, depth_limit=depth_limit,
                    buffer_kwargs=buffer_kwargs, buffer_frac=buffer_frac
                ) 
            else:

                split_polys.append(bp)
        return split_polys
    except:
        return [poly]
    
    
class BezierCurve(object):
    
    def __init__(
        self,
        nodes=None,
        degree=None,
        n_eval_points=100,
    ):
        nodes = nodes.transpose(np.argsort(np.array(nodes.shape)-2))  # hacky, to get in right orientation
        self._nodes = nodes
        self._degree = degree
        self.n_eval_points = n_eval_points
        
    @property
    def degree(self):
        if self._degree is None:
            self._degree = self.nodes.shape[1]-1
        return self._degree
    
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def _fortran_nodes(self):
        return np.asfortranarray(self.nodes)
    
    @property
    def _curve(self):
        return bezier.Curve(self._fortran_nodes, self.degree)
    
    @property
    def eval_points(self):
        return np.linspace(0, 1, self.n_eval_points)
    
    @property
    def evaluated_curve(self):
        x, y = self._curve.evaluate_multi(self.eval_points)
        return np.stack([x, y]).T
    
    @property
    def linestring(self):
        return LineString(self.evaluated_curve)
    
def vsketch_to_shapely(sketch):
    return [[LineString([Point(pt.real, pt.imag) for pt in lc]) for lc in layer] for layer in sketch.document.layers.values()]

def make_angled_line(deg=0, length=1, rad=None, center=None):
    angle = Angle(deg=deg, rad=rad).rad
    if center:
        coords = [np.array((np.cos(angle), np.sin(angle))) * length/2 * ii for ii in [-1, 1]]
        center = Point(center)
        center = np.array([center.x, center.y])
        coords = [c + center for c in coords]
        return LineString(coords)
    else:
        return LineString([(0,0), (length*np.cos(angle), length*np.sin(angle))])

    
class CenteredAngledLine(object):
    def __init__(self, center=None, deg=0, length=1, rad=None):
        self.angle = Angle(deg=deg, rad=rad).rad
        self.length = length
        if center is None:
            self.center = Point(0,0)
            
    @property
    def ls(self):
        ls = angled_line
        
    
def vsketch_to_shapely(sketch):
    return [[LineString([Point(pt.real, pt.imag) for pt in lc]) for lc in layer] for layer in sketch.document.layers.values()]
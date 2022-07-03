import shapely.geometry as sg
from shapely.geometry import box, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon, LineString
import shapely.affinity as sa
import shapely.ops as so
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas
import vpype_cli
from typing import List, Generic
from genpen import genpen as gp, utils as utils
from scipy import stats as ss
from tqdm import tqdm
from enum import Enum
import genpen

class GrowerParams(object):
    
    def __init__(
        self,
        n_pts_eval_per_iter=30,
        n_pts_add_per_iter=1,
        rads=1.,
        rotations=0.,
        n_corners=6,
        rad_func='static_rads',
        rotation_func='static_rotations',
        n_corners_func='static_n_corners',
        loss_range=(0, 100),
        rad_range=(10, 1),
        boundary_pt_dist=ss.uniform(loc=0, scale=1),
        loss_func='haussdorf_from_agg',
        pt_to_poly_func='buffer_pt',
        halt_condition_func='return_false',
        loss_threshold=1000,
        poly_transformation=None,
        
    ):
        self.n_pts_eval_per_iter = n_pts_eval_per_iter
        self.n_pts_add_per_iter = n_pts_add_per_iter
        self.static_rads = gp.make_callable(rads)
        self.static_rotations = gp.make_callable(rotations)
        self.static_n_corners = gp.make_callable(n_corners)
        self.loss_range=loss_range
        self.rad_range = rad_range
        self.rad_func = rad_func
        self.rotation_func = rotation_func
        self.n_corners_func = n_corners_func
        self.boundary_pt_dist = boundary_pt_dist
        self.loss_func = loss_func
        self.pt_to_poly_func = pt_to_poly_func
        self.halt_condition_func = halt_condition_func
        self.loss_threshold = loss_threshold
        self.poly_transformation = poly_transformation
        
    
    @property
    def _rad_func(self):
        return getattr(self, self.rad_func)
    
    def get_rad(self):
        return self._rad_func()
        
    def loss_scaled_rad(self):
        return np.interp(self.sketch.current_pt['loss'], self.loss_range, self.rad_range)
        
    # this is a silly way to do this, these should be callbacks
    @property
    def _loss_func(self):
        return getattr(self, self.loss_func)
    
    def haussdorf_from_agg(self, pt):
        return pt.hausdorff_distance(self.sketch.agg_poly)
    
    def negative_haussdorf_from_agg(self, pt):
        return pt.hausdorff_distance(self.sketch.agg_poly)
    
    def negative_distance_from_target(self, pt):
        return -pt.distance(self.sketch.target)
    
    def distance_from_target(self, pt):
        return pt.distance(self.sketch.target)
    
    @property
    def _halt_condition_func(self):
        return getattr(self, self.halt_condition_func)
    
    def return_false(self):
        return False
    
    def below_loss_threshold(self):
        return any(self.sketch.selected_pts['loss'] < self.loss_threshold)
    
    def above_loss_threshold(self):
        return any(self.sketch.selected_pts['loss'] > self.loss_threshold)
    
    @property
    def _pt_to_poly_func(self):
        return getattr(self, self.pt_to_poly_func)
    
    def buffer_pt(self, pt):
        self.sketch.current_pt = pt
        new_row = pt.copy()
        new_row['geometry'] = pt['geometry'].buffer(self.get_rad())
        return new_row
    
    def reg_poly(self, pt):
        self.sketch.current_pt = pt
        new_row = pt.copy()
        new_row['geometry'] = gp.RegPolygon(
            pt['geometry'], 
            n_corners=self.get_n_corners(), 
            rotation=self.get_rotation(), 
            radius=self.get_rad()).poly
        return new_row
    
    @property
    def _rotation_func(self):
        return getattr(self, self.rotation_func)
    
    def get_rotation(self):
        return self._rotation_func()
    
    @property
    def _n_corners_func(self):
        return getattr(self, self.n_corners_func)
    
    def get_n_corners(self):
        return self._n_corners_func()
        

class Grower(object):
    
    def __init__(
        self, 
        poly,
        params: GrowerParams,
        target=None,
    ):
        self.polys = [poly]
        self.params = params
        self.params.sketch = self
        self.new_pts = geopandas.GeoDataFrame({
            'geometry': [],
            'loss': [],
        })
        self.target = target
        self.halt_condition_satisfied = False
        
    @property
    def _p(self):
        return self.params
    
    @_p.setter
    def _p(self, _p):
        self._p = _p
    
    
    @property
    def mpoly(self):
        return gp.merge_Polygons(self.polys)
    
    @property
    def agg_poly(self):
        return so.unary_union(self.mpoly)
        
    def get_random_boundary_pts(self, n_pts=1):
        self.new_pts = geopandas.GeoDataFrame()
        self.new_pts['geometry'] = [self.agg_poly.boundary.interpolate(d, normalized=True) for d in self._p.boundary_pt_dist.rvs(n_pts)]
    
    def calc_pts_loss(self):
        self.new_pts['loss'] = self.new_pts['geometry'].apply(self._p._loss_func)
    
    def select_pts(self, n_selections=1):
        sorted_new_pts = self.new_pts.sort_values('loss')
        self.selected_pts = sorted_new_pts.iloc[:n_selections]
    
    def selected_pts_to_polys(self):
        self.new_polys = self.selected_pts.apply(self._p._pt_to_poly_func, axis=1)
        if self._p.poly_transformation is not None:
            self.new_polys['geometry'] = self.new_polys.apply(self._p.poly_transformation, axis=1)
    
    def agglomerate_polys(self):
        for ii, row in self.new_polys.iterrows():
            diff = row['geometry'].difference(self.agg_poly).buffer(1e-6)
            self.polys.append(diff)
            
    def check_halt_condition(self):
        self.halt_condition_satisfied = self._p._halt_condition_func()
        
    def grow(self, n_iters=1):
        for ii in tqdm(range(n_iters)):
            self.get_random_boundary_pts(n_pts=self._p.n_pts_eval_per_iter)
            self.calc_pts_loss()
            self.select_pts(n_selections=self._p.n_pts_add_per_iter)
            self.selected_pts_to_polys()
            self.agglomerate_polys()
            self.check_halt_condition()
            if self.halt_condition_satisfied:
                break
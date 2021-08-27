import argh
import itertools
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
from dataclasses import asdict, dataclass, field
import vsketch
import shapely.geometry as sg
from shapely.geometry import box, MultiLineString, Point, MultiPoint, Polygon, MultiPolygon, LineString
import shapely.affinity as sa
import shapely.ops as so
import matplotlib.pyplot as plt
import pandas as pd

import vpype_cli
from typing import List, Generic
from genpen import genpen as gp
from genpen.utils import Paper
from scipy import stats as ss
import geopandas
from shapely.errors import TopologicalError
import functools
import vpype
from skimage import io
from pathlib import Path

from sklearn.preprocessing import minmax_scale
from skimage import feature
from skimage import exposure

from skimage import filters
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.morphology import disk

def local_angle(dx, dy):
    """Calculate the angles between horizontal and vertical operators."""
    return np.mod(np.arctan2(dy, dx), np.pi)

if __name__ == '__main__':
    @argh.arg('-m', '--merge-tolerances', nargs='*', type=float)
    @argh.arg('-s', '--simplify-tolerances', nargs='*', type=float)
    def main(
        image_path,
        filename,
        paper_size:str = '11x17 inches',
        border:float=20,  # mm
        image_rescale_factor:float=0.25,
        smooth_disk_size:int=2,
        hist_clip_limit=0.1,
        hist_nbins=32,
        intensity_min=0.,
        intensity_max=1.,
        hatch_spacing_min=0.3,  # mm
        hatch_spacing_max=1.,  # mm
        pixel_width=0.9,  # mm
        pixel_height=0.9,  # mm
        angle_jitter='0',  # degrees
        pixel_rotation='0',  # degrees
        merge_tolerances=[0.3, 0.4, 0.5],  # mm
        simplify_tolerances=[0.2],  # mm
        savedir='/mnt/c/code/side/plotter_images/oned_outputs'
        ):
        
        
        
        # make page
        paper = Paper(paper_size)
        drawbox = paper.get_drawbox(border)
        
        # load
        img =  rgb2gray(io.imread(Path(image_path)))
        img_rescale = rescale(img, image_rescale_factor)
        
        # 
        img_renorm = exposure.equalize_adapthist(img_rescale, clip_limit=hist_clip_limit, nbins=hist_nbins)
        
        # calc dominant angle
        selem = disk(smooth_disk_size)
        filt_img = filters.rank.mean(img_renorm, selem)
        angle_farid = local_angle(filters.farid_h(filt_img), filters.farid_v(filt_img))
        
        # make pixel polys
        prms = []
        for y, row in tqdm(enumerate(img_renorm)):
            for x, intensity in enumerate(row):
                
                p = gp.centered_box(Point(x, y), width=pixel_width, height=pixel_height)
                a = np.degrees(angle_farid[y, x])
                prm = {
                    'geometry':p,
                    'x':x,
                    'y':y,
                    'raw_pixel_width':pixel_width,
                    'raw_pixel_height':pixel_height,
                    'intensity': intensity,
                    'angle':a,
                    'group': 'raw_hatch_pixel',
                    
                }
                prms.append(prm)
        raw_hatch_pixels = geopandas.GeoDataFrame(prms)
        
        #  rescale polys to fit in drawbox
        bbox = box(*raw_hatch_pixels.total_bounds)
        _, transform = gp.make_like(bbox, drawbox, return_transform=True)
        A = gp.AffineMatrix(**transform)
        scaled_hatch_pixels = raw_hatch_pixels.copy()
        scaled_hatch_pixels['geometry'] = scaled_hatch_pixels.affine_transform(A.A_flat)
        scaled_hatch_pixels['scaled_pixel_height'] = scaled_hatch_pixels['geometry'].apply(gp.get_height)
        scaled_hatch_pixels['scaled_pixel_width'] = scaled_hatch_pixels['geometry'].apply(gp.get_width)
        
        # distributions etc
        angle_jitter_gen = gp.make_callable(eval(angle_jitter))
        pixel_rotation_gen = gp.make_callable(eval(pixel_rotation))
        
        
        scaled_hatch_pixels['angle_jitter'] = angle_jitter_gen(len(scaled_hatch_pixels))
        scaled_hatch_pixels['hatch_angle'] = scaled_hatch_pixels['angle'] + scaled_hatch_pixels['angle_jitter']
        scaled_hatch_pixels['pixel_rotation'] = pixel_rotation_gen(len(scaled_hatch_pixels))
        
        example_height = scaled_hatch_pixels.loc[0, 'scaled_pixel_height']
        example_width = scaled_hatch_pixels.loc[0, 'scaled_pixel_width']
        print(f'pixel size = {example_width:.2}x{example_height:.2}mm')
        
        spacing_func = functools.partial(np.interp, xp=[intensity_min, intensity_max], fp=[hatch_spacing_max, hatch_spacing_min, ])
        scaled_hatch_pixels['spacing'] = spacing_func(1 - scaled_hatch_pixels['intensity'])
        new_rows = []
        for i, row in tqdm(scaled_hatch_pixels.iterrows(), total=len(scaled_hatch_pixels)):
            r = row.copy()
            
            p = r['geometry']
            if abs(r['pixel_rotation']) > np.finfo(float).eps:
                p = sa.rotate(p, r['pixel_rotation'])
            f = gp.hatchbox(p, spacing=r['spacing'], angle=r['hatch_angle'])
            r['geometry'] = f
            new_rows.append(r)
            
        fills = geopandas.GeoDataFrame(new_rows)
        fills = fills[fills.length > 0]
        fill_layer = gp.merge_LineStrings(fills.geometry)
        
        sk = vsketch.Vsketch()
        sk.size(paper.page_format_mm)
        sk.scale('1mm')
        sk.stroke(1)
        sk.geometry(fill_layer)
        
        sk.vpype('linesort')
        
        for tolerance in merge_tolerances:
            sk.vpype(f'linemerge --tolerance {tolerance}mm')
            
        for tolerance in simplify_tolerances:
            sk.vpype(f'linesimplify --tolerance {tolerance}mm')
            
        sk.vpype('linesort')
        
        savepath = Path(savedir).joinpath(filename).as_posix()
        sk.save(savepath)
        
    argh.dispatch_command(main)
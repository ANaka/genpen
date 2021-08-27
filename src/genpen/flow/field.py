
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





class PerlinGrid(object):
    '''slow af'''

    def __init__(self, poly, lod=4, falloff=None, noiseSeed=71, noise_scale=0.001, output_range=(0, np.pi*2)):

        self.p = poly
        self.vsk = vsketch.Vsketch()
        self.lod = lod
        self.falloff = falloff
        self.noiseSeed = noiseSeed
        self.noise_scale = noise_scale
        self.vsk.noiseSeed(self.noiseSeed)
        self.vsk.noiseDetail(lod=self.lod, falloff=self.falloff)
        self.output_range = output_range
        
    def noise(self, x, y):
        x = x * self.noise_scale
        y = y * self.noise_scale
        output = self.vsk.noise(x=x, y=y)
        return np.interp(output, [0, 1], self.output_range)
    
    def get_vector(self, p):
        angle = self.noise(p.x, p.y)
        vector = np.array((np.cos(angle), np.sin(angle)))
        return vector


class PiecewiseGrid(object):
    def __init__(self, geodf, angle_range=(0, np.pi*2)):

        self.geodf = geodf
        self.angle_range = angle_range
        
    @property
    def p(self):
        return so.unary_union(self.geodf.geometry)
        
    def get_vector(self, p):
        ind = self.geodf.contains(p)
        ind = np.flatnonzero(ind)
        assert len(ind) == 1
        ind = ind[0]
        angle = self.geodf.at[ind, 'angle']
        angle = np.radians(angle)
#         angle = np.interp(angle, [0, np.pi*2], self.angle_range)
        
        intensity = self.geodf.at[ind, 'intensity']  #not doing anything with this rn
        
        vector = np.array((np.cos(angle), np.sin(angle)))
        return vector
    
class QuantizedPiecewiseGrid(object):

    def __init__(self, geodf, xstep, ystep):

        self.geodf = geodf
        self.p = so.unary_union(self.geodf.geometry)
        (self.xbins, self.ybins), (self.gxs, self.gys) = gp.overlay_grid(self.p, xstep, ystep)
        
        
    def make_grid(self):
        dxs = np.zeros(self.gxs.shape)
        dys = np.zeros(self.gxs.shape)
        for jj, x in tqdm(enumerate(self.xbins)):
            for ii, y in enumerate(self.ybins):
                pt = Point((x,y))
                vector = self._get_vector(pt)
                dxs[ii, jj] = vector[0]
                dys[ii, jj] = vector[1]
                
        self.dxs = dxs
        self.dys = dys
        
    def get_containing_geom(self, p):
        ind = self.geodf.contains(p)
        ind = np.flatnonzero(ind)
        assert len(ind) <= 1, print(ind)
        ind = ind[0]
        return self.geodf.loc[ind, :]
        
    def _get_vector(self, pt):
        try:
            geom_row = self.get_containing_geom(pt)
            angle = geom_row['angle']
            angle = np.radians(angle)
            magnitude = geom_row['magnitude']  #not doing anything with this rn
            vector = np.array((np.cos(angle), np.sin(angle))) * magnitude
        except:
            vector = np.array((0,0))
        return vector
    
    def get_vector(self, pt):
        
        xind, yind = self.get_closest_bins(pt.x, pt.y)
        dx = self.dxs[yind, xind]
        dy = self.dys[yind, xind]
        vector = np.array((dx, dy))
        return np.array((dx, dy))
        
    def get_closest_bins(self, x, y):
        xind = np.argmin(abs(self.xbins-x))
        yind = np.argmin(abs(self.ybins-y))
        return xind, yind
    
class NoisyQuantizedPiecewiseGrid(QuantizedPiecewiseGrid):
    
    def __init__(self, 
                 geodf, xstep, ystep, lod=4, 
                 falloff=None, noiseSeed=71, noise_scale=0.001,
                 output_range=(0, np.pi*2),
                 noise_mult=1,
                 verbose=False,
                ):
        
        self.geodf = geodf
        self.p = so.unary_union(self.geodf.geometry)
        (self.xbins, self.ybins), (self.gxs, self.gys) = gp.overlay_grid(self.p, xstep, ystep)
        self.pg = PerlinGrid(
            poly=self.p, 
            lod=lod, 
            falloff=falloff, 
            noiseSeed=noiseSeed, 
            noise_scale=noise_scale, 
            output_range=output_range
        )
        self.noise_mult = noise_mult
        self.verbose = verbose
        
    def get_noise_vec(self, pt):
        n = self.pg.noise(pt.x, pt.y)
        return np.array((np.cos(n), np.sin(n)))
        
    def get_vector(self, pt, noise_mult=1):
        xind, yind = self.get_closest_bins(pt.x, pt.y)
        dx = self.dxs[yind, xind]
        dy = self.dys[yind, xind]
        vector = np.array((dx, dy)) / (1 + self.noise_mult)
        noise = (self.get_noise_vec(pt)* self.noise_mult) / (1 + self.noise_mult)
        if self.verbose:
            print(f'grid: {vector}')
            print(f'noise: {noise}')
        return vector + noise

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

class VectorParticle(object):

    def __init__(self, pos, grid, vector=None, momentum_factor=0., stepsize=1):
        self.pos = pos
        self.grid = grid
        self.stepsize = stepsize
        self.momentum_factor = momentum_factor
        self.n_step = 0
        self.pts = [self.pos]
        self.in_bounds = True
        if vector is None:
            vector = np.array([0, 0])
        self.vector = vector

    @property
    def x(self):
        return self.pos.x
    
    @property
    def y(self):
        return self.pos.y
    
    @property
    def xy(self):
        return np.array([self.x, self.y])

    @property
    def line(self):
        return LineString(self.pts)
    
    def update_vector(self):
        new_vector = self.grid.get_vector(self.pos)
        updated = (self.vector * self.momentum_factor + new_vector) / (1 + self.momentum_factor)
        self.vector = updated
        
    def check_if_in_bounds(self):
        self.in_bounds = self.grid.p.contains(self.pos)

    def calc_step(self):
        self.update_vector()
        self.dx = self.vector[0] * self.stepsize
        self.dy = self.vector[1] * self.stepsize


    def step(self):
        self.check_if_in_bounds()
        if self.in_bounds:
            self.calc_step()
            self.pos = sa.translate(self.pos, xoff=self.dx, yoff=self.dy)
            self.pts.append(self.pos)
            

# Cell
class Particle(object):

    def __init__(self, pos, grid, stepsize=1):
        self.pos = Point(pos)
        self.grid = grid
        self.stepsize = stepsize
        self.n_step = 0
        self.pts = [self.pos]
        self.in_bounds = True

    @property
    def x(self):
        return self.pos.x
    
    @property
    def y(self):
        return self.pos.y
    
    @property
    def xy(self):
        return np.array([self.x, self.y])

    @property
    def line(self):
        return LineString(self.pts)
    
    def get_closest_bins(self):
        self.xind = np.argmin(abs(self.grid.xbins-self.x))
        self.yind = np.argmin(abs(self.grid.ybins-self.y))

    def get_angle(self):
        self.a = self.grid.a[self.yind, self.xind]

    def check_if_in_bounds(self):
        self.in_bounds = self.grid.p.contains(self.pos)

    def calc_step(self):
        self.get_closest_bins()
        self.get_angle()
        self.dx = np.cos(self.a) * self.stepsize
        self.dy = np.sin(self.a) * self.stepsize


    def step(self):
        self.check_if_in_bounds()
        if self.in_bounds:
            self.calc_step()
            self.pos = sa.translate(self.pos, xoff=self.dx, yoff=self.dy)
            self.pts.append(self.pos)

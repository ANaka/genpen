import bezier
from dataclasses import asdict, dataclass
import numpy as np
import shapely.geometry as sg
import shapely.affinity as sa
import shapely.ops as so

from shapely import speedups
from tqdm import tqdm

class DistanceConverter(object):

    def __init__(self, d, unit):
        setattr(self, unit, d)

    @property
    def inches(self):
        return self._inches

    @inches.setter
    def inches(self, inches):
        self._inches = inches
        self._mm = 25.4 * inches

    @property
    def mm(self):
        return self._mm

    @mm.setter
    def mm(self, d):
        self._mm = d
        self._inches = d / 25.4





class Paper(object):
    
    def __init__(
            self,
            size:str='11x14 inches',
            
        ):
        standard_sizes = {
            'letter':'8.5x11 inches',
            'A3': '11.7x16.5 inches',
            'A4': '8.3x11.7 inches',
            'A2': '17.01x23.42 inches' # 432x594mm
        }
        
        std_size = standard_sizes.get(size, None)
        if std_size is not None:
            size = std_size
        _x, _y, _units = self.parse_size_string(size)
        
        self.x = DistanceConverter(_x, _units)
        self.y = DistanceConverter(_y, _units)
        
    
    
        
    @property
    def page_format_mm(self):
        return f'{self.x.mm}mmx{self.y.mm}mm'
    
    @staticmethod
    def parse_size_string(size_string):
        size, units = size_string.split(' ')
        x,y = size.split('x')
        return float(x), float(y), units
        
    def get_drawbox(
        self, 
        border:float=0,  # mm
        xborder=None, 
        yborder=None):
        
        if xborder is None:
            xborder = border
        if yborder is None:
            yborder = border
            
        return sg.box(xborder, yborder, self.x.mm-xborder, self.y.mm-yborder)
    
    

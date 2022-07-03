from xml.etree import ElementTree
from typing import List
from copy import deepcopy
import click
import numpy as np
from mecode import G
import vsketch
from tqdm import tqdm
from genpen import genpen as gp


def pen_down(g, down_dist=-20):
    down_dist = np.clip(down_dist, 0, np.inf)
    g.move(x=0, y=0, z=down_dist)
    
def pen_up(g, up_dist=20):
    up_dist = np.clip(up_dist, 0, up_dist)
    g.move(x=0, y=0, z=up_dist)
    
@click.command()
@click.argument('svg_file')
@click.argument('gcode_file')
@click.option('--down_loc', default=0)
@click.option('--up_loc', default=20)
def main(svg_file, gcode_file, down_loc, up_loc):
    g = G(outfile=gcode_file) # init
    sk = vsketch.Vsketch() # init
    sk.vpype(f'read {svg_file}')
    output = gp.vsketch_to_shapely(sk)
    g.abs_move(x=0, y=0, z=up_loc)
    for layer in output:
        for ls in tqdm(layer):
            coords = list(ls.coords)
            n_coords = len(coords)
            current_coord = coords[0]
            g.abs_move(x=current_coord[0], y=current_coord[1], z=up_loc)
            g.abs_move(x=current_coord[0], y=current_coord[1], z=down_loc)
            for current_coord in coords[1:]:
                g.abs_move(x=current_coord[0], y=current_coord[1], z=0)
            g.abs_move(x=current_coord[0], y=current_coord[1], z=up_loc)
            
if __name__ == '__main__':
    main()
import signal
from pov import pov
from pathlib import Path
import fn
import time
import vpype
from pyaxidraw import axidraw 
from contextlib import closing
from genpen import genpen as gp
from tqdm import tqdm

class GracefulExiter():
    # definitely jacked this from somewhere on stack overflow
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("exit flag set to True (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state
    
    
class AxiCam(object):
    
    def __init__(
        self, 
        svg_path=None,
        image_savedir=None,
        plot_id=None,
        cam=None,
        ):
        if plot_id == None:
            plot_id = fn.get_current_plot_id()
        self.plot_id = plot_id
        
        if svg_path == None:
            svg_path = Path(gp.SVG_SAVEDIR).joinpath(plot_id).with_suffix('.svg')
        self.svg_path = svg_path    
        self.ad = axidraw.AxiDraw()
        
        self.ad.plot_setup(self.svg_path)
        self.ad.options.mode = "layers"
        self.ad.options.units = 2
        self.ad.update()
        
        self.doc = vpype.read_multilayer_svg(self.svg_path, 0.1)
        self.image_savedir = image_savedir
        self.cam = cam
        
    @property
    def n_layers(self):
        return len(self.doc.layers)
        
    def plot_layer(self, cam, layer_number, wait_time=0.):
        self.ad.options.layer = layer_number
        self.ad.plot_run()
        time.sleep(wait_time)
        cam.save_image()
        
    def init_cam(self):
        if self.cam is not None:
            try:
                self.cam.close()
            except:
                pass
        
        self.cam = pov.Camera(savedir=self.image_savedir)
        return self.cam
        
    
        
    def plot_layers(self, prog_bar=True, wait_times=0., start_layer=0):
        wait_times = gp.make_callable(wait_times)
        iterator = range(start_layer, self.n_layers)
        if prog_bar:
            iterator = tqdm(iterator)
        
        flag = GracefulExiter()
        self.init_cam()
        for layer_number in iterator:
            wait_time = wait_times()
            self.plot_layer(cam=self.cam, layer_number=layer_number, wait_time=wait_time)
            if flag.exit():
                self.cam.close()
                break
            
        self.cam.save_image()
        
        self.cam.close()
        
    def toggle_pen(self):
        self.ad.options.mode = 'toggle'
        self.ad.plot_run()
        self.ad.options.mode = 'layers'
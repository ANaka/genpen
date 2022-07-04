from torch import Tensor
from torch import nn
from PIL import Image
import attr
import pydiffvg as dg
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import numpy as np
import clip


class Canvas(object):
    
    def __init__(
        self, 
        width=256, 
        height=256, 
        draw_white_background=False,
        multiplier=1,
        ):
        self.width = width
        self.height = height
        self.draw_white_background = draw_white_background
        self.multiplier = multiplier
        
    def get_scene_args(self, shapes, shape_groups):
        return dg.RenderFunction.serialize_scene(
            self.width * self.multiplier,
            self.height * self.multiplier,
            shapes, shape_groups)

    def render(self, shapes, shape_groups, as_pil: bool = False):
        # Rasterize the image.
        scene_args = self.get_scene_args(shapes, shape_groups)
        img = dg.RenderFunction.apply(
            self.width * self.multiplier,
            self.height * self.multiplier,
            2, 2, 0, None, *scene_args)
        if self.draw_white_background:
            w, h = img.shape[0], img.shape[1]
            img = img[:, :, 3:4] * img[:, :, :3] + (
                torch.ones(w, h, 3, device=dg.get_device()) * (1-img[:, :, 3:4]))
        else:
            img = img[:, :, :3]
        img = img.unsqueeze(0).permute(0, 3, 1, 2)
        
        if as_pil:
            return to_pil(img)
        else:
            img = img
            return img


def to_pil(img):
    return TF.to_pil_image(img.detach().cpu().squeeze())

    
def LazyRelu(input_size):
    return nn.Sequential(
        nn.LazyLinear(input_size),
        nn.ReLU(),
    )
    
def LazyHardsigmoid(input_size):
    return nn.Sequential(
        nn.LazyLinear(input_size),
        nn.Hardsigmoid(),
    )
    
def LazySigmoid(input_size):
    return nn.Sequential(
        nn.LazyLinear(input_size),
        nn.Sigmoid(),
    )
    
def LazyTanh(input_size):
    return nn.Sequential(
        nn.LazyLinear(input_size),
        nn.Tanh(),
    )

class CanvasNet(nn.Module):
    
    def __init__(
        self, 
        backbone, 
        shapes_head,
        shape_groups_head,
        canvas,
        ):
        super().__init__()
        self.backbone = backbone
        self.shapes_head = shapes_head
        self.shape_groups_head = shape_groups_head
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seed_tensor = nn.Parameter(torch.tensor([0.5])).to(self.device)
        self.canvas = canvas
        
    def forward(self, input=None):
        input = self.seed_tensor if input is None else input
        x = self.backbone.forward(input)
        shapes, shapes_tensors = self.shapes_head.forward(x)
        shape_groups, shape_groups_tensors = self.shape_groups_head.forward(x)
        tensors = {
            'x': x,
            'shapes_tensors': shapes_tensors,
            'shape_groups_tensors': shape_groups_tensors,
        }
        img = self.canvas.render(shapes, shape_groups)
        return shapes, shape_groups, tensors, img
    
class CanvasNIMANet(nn.Module):
    
    def __init__(self, canvas_net, nima):
        super().__init__()
        self.canvas_net = canvas_net
        self.nima = nima
        
    def forward(self, input=None):
        return self.canvas_net.forward(input)
    
    def nima_loss(self, img):
        scores = self.nima(img)
        weighted_votes = torch.arange(10, dtype=torch.float, device=img.device) + 1
        return -torch.matmul(scores, weighted_votes)
    
class CanvasCLIPNet(nn.Module):
    
    def __init__(self, canvas_net, perceptor):
        super().__init__()
        self.canvas_net = canvas_net
        self.perceptor = perceptor
        self.text_targets = []
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, input=None):
        return self.canvas_net.forward(input)
    
    def encode_text(self, text):
        return self.perceptor.encode_text(clip.tokenize(text).to(self.device)).float()
    
    def set_text_prompts(self, text_prompts):
        self.text_targets = [self.encode_text(s) for s in text_prompts]
        
    def encode_canvas_img(self, img):
        return self.perceptor.encode_image(self.normalize(img)).float()
    
    def get_clip_losses(self, img):
        z = self.encode_canvas_img(img)
        return [F.mse_loss(z, t) for t in self.text_targets]
    
        
        
class ShapeMaker(nn.Module):
    
    def __init__(
        self, 
        shape_class,
        n_shapes,
        base_id=None,
        param_transforms=None,
        ):
        super().__init__()
        self.shape_class = shape_class
        self.base_id = np.random.randint(1e5) if base_id is None else base_id
        self.n_shapes = n_shapes
        self.ids = np.arange(n_shapes) + self.base_id
        param_transforms = {} if param_transforms is None else param_transforms
        self.param_transforms = nn.ModuleDict(param_transforms)
        
        
        
    def forward(self, params:dict):
        
        shapes = []
        
        
        for param_name, transform in self.param_transforms.items():
            params[param_name] = transform.forward(params[param_name])
            
        for ii in range(self.n_shapes):
            param_set = {key: value[ii] for key, value in params.items()}
            param_set['id'] = self.ids[ii]
            shapes.append(self.shape_class(**param_set))
            
        return shapes, params
    
class ShapeGroupMaker(nn.Module):
    
    def __init__(self, shape_ids, param_transforms=None):
        super().__init__()
        self.shape_ids = torch.tensor(shape_ids)
        self.param_transforms = {} if param_transforms is None else param_transforms
        
    
    def forward(self, params):
        
        for param_name, transform in self.param_transforms.items():
            params[param_name] = transform(params[param_name])
            
        return [dg.ShapeGroup(
            shape_ids = self.shape_ids,
            **params)], params
        
class ParamHead(nn.Module):
    
    def __init__(self, nets:dict):
        super().__init__()
        self.nets = nn.ModuleDict(nets)
        
        
    def forward(self, x):
        return {key: net(x) for key, net in self.nets.items()}
        
        
def init_weights(m, gain=3):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.01)
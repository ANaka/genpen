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

@attr.define()
class CircleP(object):
    center: Tensor = Tensor((0., 0.))
    radius: Tensor = Tensor((1.,))
    stroke_width: Tensor = Tensor((1.,))
    color: Tensor = Tensor((1., 1., 1., 1.))
    
    def requires_grad(self, requires_grad=True):
        self.center.requires_grad = requires_grad
        self.radius.requires_grad = requires_grad
        self.stroke_width.requires_grad = requires_grad
        self.color.requires_grad = requires_grad
        return self
    

CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800
DRAW_WHITE_BACKGROUND = False

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
        img = img.unsqueeze(0)
        
        if as_pil:
            return to_pil(img)
        else:
            img = img.permute(0, 3, 1, 2)
            return img

def render_circles(cc, multiplier=1):
    """Render circles to image.

    Args:
      centers: points defining the lines
      stroke_widths: the widths of the lines
      all_colors: line colours
      multiplier: scale factor to enlarge drawing

    Returns:
      image with lines drawn
    """
    # Store `centers` definitions as shapes, colours and widths.
    shapes = []
    shape_groups = []
    for c in cc:
      center = c.center.contiguous().cpu()
      radius = c.radius.cpu()
      width = c.stroke_width.cpu()
      color = c.color.cpu()
      path = dg.Circle(
          radius=radius*multiplier, 
          center=center * multiplier,
          stroke_width=width * multiplier
          )
      shapes.append(path)
      path_group = dg.ShapeGroup(
          shape_ids=torch.tensor([len(shapes) - 1]),
          fill_color=None,
          stroke_color=color)
      shape_groups.append(path_group)

    # Rasterize the image.
    scene_args = dg.RenderFunction.serialize_scene(
        CANVAS_WIDTH * multiplier,
        CANVAS_HEIGHT * multiplier,
        shapes, shape_groups)
    img = dg.RenderFunction.apply(
        CANVAS_WIDTH * multiplier,
        CANVAS_HEIGHT * multiplier,
        2, 2, 0, None, *scene_args)
    if DRAW_WHITE_BACKGROUND:
      w, h = img.shape[0], img.shape[1]
      img = img[:, :, 3:4] * img[:, :, :3] + (
          torch.ones(w, h, 3, device=dg.get_device()) * (1-img[:, :, 3:4]))
    else:
      img = img[:, :, :3]
    img = img.unsqueeze(0)

    return img

def to_pil(img):
    return TF.to_pil_image(img.detach().cpu().squeeze().permute(2, 0, 1))

class CircleCollection(nn.Module):
    
    def __init__(self, centers, radii, stroke_widths, colors, ids):
        super().__init__()
        self.centers = centers
        self.radii = radii
        self.stroke_widths = stroke_widths
        self.colors = colors
        self.ids = ids
        
    @classmethod
    def from_arrays(cls, centers, radii, stroke_widths, colors, ids):
        centers = torch.tensor(centers, requires_grad=True)
        radii = torch.tensor(radii, requires_grad=True)
        stroke_widths = torch.tensor(stroke_widths, requires_grad=True)
        colors = torch.tensor(colors, requires_grad=True)
        ids = torch.tensor(ids)
        return cls(centers, radii, stroke_widths, colors, ids)
        
    @property
    def n_circles(self):
        return self.centers.shape[0]
    
    def requires_grad(self, requires_grad=True):
        self.centers.requires_grad = requires_grad
        self.radii.requires_grad = requires_grad
        self.stroke_widths.requires_grad = requires_grad
        self.colors.requires_grad = requires_grad
        return self
    
    def __getitem__(self, ii):
        return CircleP(
            self.centers[ii], 
            self.radii[ii], 
            self.stroke_widths[ii], 
            self.colors[ii],
            self.ids[ii],
            )
    
    def __iter__(self):
        for ii in range(self.n_circles):
            yield self[ii]
        
    def to_circle_ps(self):
        circle_ps = []
        for ii in range(self.centers.shape[0]):
            circle_ps.append(CircleP(
                center=self.centers[ii],
                radius=self.radii[ii],
                stroke_width=self.stroke_widths[ii],
                color=self.colors[ii]
            ))
        return circle_ps
    
    def get_shapes_and_shapegroups(self):
        shapes = []
        shape_groups = []
        for ii, param_set in enumerate(self):
            path = dg.Circle(
                radius=param_set.radius.cpu(), 
                center=param_set.center.contiguous().cpu(),
                stroke_width=param_set.stroke_width.cpu(),
                
                )
            shapes.append(path)
            path_group = dg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=param_set.color.cpu())
            shape_groups.append(path_group)
    
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
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class CircleNet(nn.Module):
    
    
    def __init__(
        self, 
        n_circles,
        x_range,
        y_range,
        rad_range,
        init_centers=None,
        init_radii=None,
        ):
        super().__init__()
        self.n_circles = n_circles
        self.xmin, self.xmax = x_range
        self.ymin, self.ymax = y_range
        self.radmin, self.radmax = rad_range
        
        if init_centers is None:
            xs = torch.rand(n_circles, 1) * (self.xmax - self.xmin) + self.xmin
            ys = torch.rand(n_circles, 1) * (self.ymax - self.ymin) + self.ymin
            init_centers = torch.cat([xs, ys], axis=1)
        self.init_centers = init_centers
        
        if init_radii is None:
            init_radii = torch.rand(n_circles) * (self.radmax - self.radmin) + self.radmin
        self.init_radii = init_radii
        
        self.net =nn.Sequential(
            nn.Linear(0, 128),
            nn.ReLU(),
            LazyRelu(128),
            LazyRelu(128),
            LazyRelu(128),
        )
        
        
        self.x_outputs = LazyTanh(self.n_circles)
        self.y_outputs = LazyTanh(self.n_circles)
        self.radius_outputs = LazyTanh(self.n_circles)
        
        
    def initialize_weights(self):
        self.forward()
        self.net.apply(init_weights)
        self.x_outputs.apply(init_weights)
        self.y_outputs.apply(init_weights)
        self.radius_outputs.apply(init_weights)
        
    def get_net_output(self):
        return self.net(Tensor())
        
    def get_radii(self, x):
        return self.radius_outputs(x) * (self.radmax - self.radmin) * 0.5 + self.init_radii
    
    def get_xs(self, x):
        return self.x_outputs(x).unsqueeze(1) * (self.xmax - self.xmin) * 0.5
    
    def get_ys(self, x):
        return self.y_outputs(x).unsqueeze(1) * (self.ymax - self.ymin) * 0.5 
    
    def get_centers(self, x):
        return torch.cat([self.get_xs(x), self.get_ys(x)], axis=1) + self.init_centers
    
    def forward(self):
        x = self.get_net_output()
        centers = self.get_centers(x)
        radii = self.get_radii(x)
        colors = torch.ones((self.n_circles,4))
        stroke_widths = torch.ones((self.n_circles,1)) * 0.4

        cc = CircleCollection(centers=centers, radii=radii, stroke_widths=stroke_widths, colors=colors)
        return cc
    

    
    
def train(loss_weights=None):
    
    cc = model.forward()
    
    if loss_weights is None:
        loss_weights = {
            'big_rad_loss': 1,
            'overlap_loss': 1,
            'distance_loss': 1,
            'out_of_bounds_loss': 1,
            
        }
    optim.zero_grad()
    losses = {}
    overlap_buffer = 0

    # desired rad size
    # target_rad_loss = (target_radius - cc.radii).pow(2).sum()
    
    # big rad 
    losses['big_rad_loss'] = (-cc.radii).exp().sum()

    # prevent overlap
    dists = torch.cdist(cc.centers, cc.centers)
    upper_dists = torch.triu(dists)
    nonzero_inds = torch.nonzero(upper_dists, as_tuple=True)
    dists[nonzero_inds]
    summed_rads = cc.radii.unsqueeze(0).T + cc.radii.unsqueeze(0)
    edge_to_edge_dists = dists[nonzero_inds] - summed_rads[nonzero_inds] - overlap_buffer
    losses['overlap_loss'] = F.relu(-edge_to_edge_dists).pow(3).sum()
    losses['distance_loss'] = edge_to_edge_dists.pow(2).sum()

    # out of bounds
    xs = cc.centers[:, 0]
    ys = cc.centers[:, 1]
    left_edge_dist = F.relu(-((xs - cc.radii) - xmin))
    right_edge_dist = F.relu(-((xmax - (xs + cc.radii))))
    bottom_edge_dist = F.relu(-((ys - cc.radii) - ymin))
    top_edge_dist = F.relu(-(ymax - (ys + cc.radii)))
    losses['out_of_bounds_loss'] = (left_edge_dist + right_edge_dist + bottom_edge_dist + top_edge_dist).pow(2).sum()
    
    #maximize skewnewss
    # losses['skewness_loss'] = -(moment(cc.radii, n=3).pow(2))
    # losses['variance_loss'] = -(moment(cc.radii, n=2).pow(2))

    scaled_losses = {key: loss*loss_weights[key] for key, loss in losses.items()}
    loss = sum(scaled_losses.values())

    loss.backward()
    optim.step()
    
    scaled_losses['loss'] = loss
    return losses, scaled_losses, cc

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
        self.seed_tensor = nn.Parameter(torch.tensor([0.]))
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
        init_params = None,
        ):
        super().__init__()
        self.shape_class = shape_class
        self.base_id = np.random.randint(1e5) if base_id is None else base_id
        self.n_shapes = n_shapes
        self.ids = np.arange(n_shapes) + self.base_id
        self.param_transforms = {} if param_transforms is None else param_transforms
        self.init_params = {} if init_params is None else init_params
        
        
        
    def forward(self, params:dict):
        
        shapes = []
        
        
        for param_name, transform in self.param_transforms.items():
            params[param_name] = transform(params[param_name])
            
            if self.init_params:
                params[param_name] += self.transformed_init_params[param_name]
            
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
        
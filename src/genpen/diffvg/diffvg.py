from torch import Tensor
from torch import nn
from PIL import Image
import attr
import pydiffvg as dg
import torch
from torchvision import transforms, models
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
        
        return dict(
            shapes=shapes, 
            shape_groups=shape_groups, 
            tensors=tensors, 
            img=img)
    
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
        
        
def init_weights(m, gain=0.3):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.01)
        
        
    
class EdgeToEdgeDistanceLoss(nn.Module):
    
    def __init__(self, safe_distance=0.):
        super().__init__()
        self.safe_distance = safe_distance
        
    def forward(self, outputs):
        tensors = outputs['tensors']
        shapes_tensors = tensors['shapes_tensors']
        dists = torch.cdist(shapes_tensors['center'], shapes_tensors['center'])
        upper_dists = torch.triu(dists)
        nonzero_inds = torch.nonzero(upper_dists, as_tuple=True)
        dists[nonzero_inds]
        summed_rads = torch.atleast_2d(shapes_tensors['radius']).T + torch.atleast_2d(shapes_tensors['radius'])
        edge_to_edge_dists = dists[nonzero_inds] - summed_rads[nonzero_inds]
        if self.safe_distance is None:
            return torch.exp(-edge_to_edge_dists).mean()
        else:
            return F.relu(self.safe_distance - edge_to_edge_dists).mean()
    
class OutOfBoundsLoss(nn.Module):
    
    def __init__(
        self,
        left,
        right,
        top,
        bottom,
        ):
        super().__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
    
    def forward(self, outputs):
        tensors = outputs['tensors']
        centers = tensors['shapes_tensors']['center']
        xs = centers[:, 0]
        ys = centers[:, 1]
        radii = tensors['shapes_tensors']['radius']
        
        left_edge_dist = torch.exp(-((xs -radii) - self.left))
        right_edge_dist = torch.exp(-((self.right - (xs + radii))))
        bottom_edge_dist = torch.exp(-((ys - radii) - self.bottom))
        top_edge_dist = torch.exp(-(self.top - (ys + radii)))
        return (left_edge_dist + right_edge_dist + bottom_edge_dist + top_edge_dist).mean()
    
    
class TargetImageLoss(nn.Module):
    
    def __init__(self, target_img):
        super().__init__()
        self.target_img = target_img.detach()
        
    def forward(self, outputs):
        img = outputs['img']
        return ((self.target_img - img) ** 2).mean()
    
class NIMALoss(nn.Module):
    
    def __init__(
        self, 
        weight_path='/home/naka/code/side/ML-Aesthetics-NIMA/weights/dense121_all.pt',
        num_classes=10,
        ):
        
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Sequential(
            nn.Linear(num_ftrs,num_classes),
            nn.Softmax(1)
        )   

        # Send the model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)
        for param in model_ft.parameters():
            param.requires_grad = False
        self.model_ft = model_ft
    
    def forward(self, outputs):
        img = outputs['img']
        scores = self.model_ft(img)
        weighted_votes = torch.arange(10, dtype=torch.float, device=img.device) + 1
        return -torch.matmul(scores, weighted_votes)
    
    
class CLIPLoss(nn.Module):
    
    def __init__(self, model="ViT-B/32"):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        perceptor, preprocess = clip.load(model, device=device)
        for param in perceptor.parameters():
            param.requires_grad = False
        self.perceptor = perceptor
        self.text_targets = []
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, outputs):
        img = outputs['img']
        z = self.perceptor.encode_image(self.normalize(img)).float()
        return sum([F.mse_loss(z, t) for t in self.text_targets])
    
    def encode_text(self, text):
        return self.perceptor.encode_text(clip.tokenize(text).to(self.device)).float()
    
    def set_text_prompts(self, text_prompts):
        self.text_targets = [self.encode_text(s) for s in text_prompts]
        
        

class MakeCutouts(nn.Module):
    def __init__(self, cutout_params, cut_size, cutn, cut_pow=1., noise_fac=0.1, use_pooling=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.cutout_params = cutout_params
        self.cutout_params['random_crop']['size'] = (self.cut_size,self.cut_size)
        self.cutout_params['random_resized_crop']['size'] = (self.cut_size,self.cut_size)
        self.use_pooling = use_pooling
        # Pick your own augments & their order
        self.augment_list = []
        for aug_name, aug_settings in self.cutout_params.items():
            if aug_settings['use']:
                params = {key: value for key, value in aug_settings.items() if key != 'use'}
                func_name = ''.join([c.capitalize() for c in aug_name.split('_')])
                aug = getattr(K, func_name)(**params)
                self.augment_list.append(aug)
            
        # print(augment_list)
        
        self.augs = nn.Sequential(*self.augment_list)

        '''
        self.augs = nn.Sequential(
            # Original:
            # K.RandomHorizontalFlip(p=0.5),
            # K.RandomVerticalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            # K.RandomSharpness(0.3,p=0.4),
            # K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5),
            # K.RandomCrop(size=(self.cut_size,self.cut_size), p=0.5), 
            # Updated colab:
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),        
            )
        '''
            
        self.noise_fac = noise_fac
        # self.noise_fac = False
        
        # Pooling
        if use_pooling:
            self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
            self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            if self.use_pooling:
                cutout = (self.av_pool(input) + self.max_pool(input))/2
                cutouts.append(cutout)
                
            else:
                size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            # cutout = transforms.Resize(size=(self.cut_size, self.cut_size))(input)
            
            
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
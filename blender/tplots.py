import bpy
from mathutils import *
D = bpy.data
C = bpy.context



def create_sphere(color, name, pos=(0,0,0), radius=1):
    """Creates a sphere in blender with a glossy surface"""
    mat = D.materials.new(name)
    mat.diffuse_color = color
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = (1,1,1)
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = 1
    mat.ambient = 1
    mat.emit = 0
    mat.specular_hardness = 50
    mat.use_transparency = True
    mat.transparency_method = 'MASK'
    mat.raytrace_mirror.use = True
    mat.raytrace_mirror.reflect_factor = 0.5
    mat.raytrace_mirror.fresnel = 1
    mat.raytrace_mirror.fresnel_factor = 2
    mat.raytrace_mirror.depth = 3
    mat.raytrace_mirror.gloss_factor = 1
    mat.raytrace_mirror.gloss_threshold = 0.0
    mat.raytrace_mirror.gloss_samples = 0
    mat.raytrace_mirror.gloss_anisotropic = 1
    mat.use_raytrace = False
    mat.use_shadows = True
    mat.use_cubemap = True
    mat.use_sky = False
    mat.use_transparent_shadows = True
    mat.use_ray_shadow_bias = False
    mat.use_cast_buffer_shadows = True
    mat.use_cast_approximate = True
    mat.use_cast_shadows_only = False
    mat.use_cast_buffer_shadows_approx = False
    mat.use_cast_approximate = False
    mat.use_cast_shadows_only = False
    mat.use_cast_buffer_shadows_approx = False
    mat.use_cast_approximate = False
    mat.use

def main():
    create_sphere(pos=(0,0,0), radius=1, color='red', name='test')
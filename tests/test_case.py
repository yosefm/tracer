import unittest
import numpy as N
import math

from tracer_engine import TracerEngine
from ray_bundle import solar_disk_bundle
from flat_surface import FlatSurface
from spatial_geometry import general_axis_rotation
from sphere_surface import SphereSurface
from boundary_shape import BoundarySphere
from receiver import Receiver
from assembly import Assembly
from object import AssembledObject
from spatial_geometry import generate_transform
from paraboloid import Paraboloid
from ray_bundle import RayBundle

def test_case():
    w = 2.
    h = 2.
    a=5
    b=5
    bound_radius = 3.
    bound_center = N.array([0,0.,0])
    transform1 = generate_transform(N.array([1.,0,0]), 3*N.pi/4, N.c_[[0,5.,5]])
    transform2 = generate_transform(N.array([1.,0,0]), -N.pi/4, N.c_[[0,0.,0]])  
    num_rays = 500
    center = N.c_[[0,0.,0.]]
    x = 1/(math.sqrt(2))
    direction = N.array([0,-x,-x])
    radius_sun = 2. 
    ang_range = N.pi/10
    iterate = 2
    min_energy = 0.05

    assembly = Assembly()
    surface1 = Receiver(width=w, height=h)
    rot1 = general_axis_rotation(N.array([0,0,1]), N.pi/2)
    surface2 = FlatSurface(location=N.array([0,2,1.]), rotation=rot1, width=w, height=h)
    surface3 = FlatSurface(location=N.array([0,-2,1.]), rotation=rot1, width=w, height=h)
    object1 = AssembledObject()
    object1.add_surface(surface1)
    object1.add_surface(surface2)
    object1.add_surface(surface3)
    
    surface4 = Paraboloid(a=a, b=b)
    bound1 = BoundarySphere(location=bound_center, radius=bound_radius)
    object2 = AssembledObject()
    object2.add_surface(surface4)
    object2.add_boundary(bound1)
    
    assembly.add_object(object1, transform1)
    assembly.add_object(object2, transform2)
    
    sun = solar_disk_bundle(num_rays, center, direction, radius_sun, ang_range)
    sun.set_energy(N.ones(num_rays))
    sun.set_ref_index(N.ones(num_rays))
    
    x = 1/(math.sqrt(2))
    dir = N.c_[[0,x,x],[0,-x,-x]]
    position = N.c_[[0,1,1],[0,1,1]]
    '''
    sun = RayBundle()
    sun.set_vertices(position)
    sun.set_directions(dir)
    sun.set_ref_index(N.r_[[1,1]])
    sun.set_energy(N.r_[[1,1]])
    '''
    engine = TracerEngine(assembly)
    engine.ray_tracer(sun, iterate, min_energy)
    surface1.plot_energy()

test_case()

import numpy as N
import math

from tracer.tracer_engine import TracerEngine
from tracer.ray_bundle import RayBundle, solar_disk_bundle
from tracer.flat_surface import FlatSurface
from tracer.spatial_geometry import general_axis_rotation, generate_transform
from tracer.sphere_surface import SphereSurface
from tracer.boundary_shape import BoundarySphere
from tracer.receiver import Receiver
from tracer.assembly import Assembly
from tracer.object import AssembledObject
from tracer.paraboloid import Paraboloid

def test_case(focus):
    w = 10.
    h = 10.
    a = 2*N.sqrt(focus)
    b = a
    
    bound_radius = 3.
    bound_center = N.array([0,0.,0])
    transform1 = generate_transform(N.array([1.,0,0]), 3*N.pi/4, N.c_[[0, 2.5, 2.5]])
    transform2 = generate_transform(N.array([1.,0,0]), -N.pi/4, N.c_[[0,0.,0]])  
    num_rays = 10000
    center = N.c_[[0, 2., 2.]]
    x = -1/(math.sqrt(2))
    direction = N.array([0,x,x])
    radius_sun = 5. 
    ang_range = 0.0005
    iterate = 2
    min_energy = 0.05

    assembly = Assembly()
    surface1 = Receiver(width=w, height=h)
    rot1 = general_axis_rotation(N.array([0,0,1]), N.pi/2)
    surface2 = FlatSurface(location=N.array([0,2,1.]), rotation=rot1, width=w, height=h)
    surface3 = FlatSurface(location=N.array([0,-2,1.]), rotation=rot1, width=w, height=h)
    object1 = AssembledObject()
    object1.add_surface(surface1)
    #object1.add_surface(surface2)
    #object1.add_surface(surface3)
    
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
    
    engine = TracerEngine(assembly)
    engine.ray_tracer(sun, iterate, min_energy)
    surface1.plot_energy()

if __name__ == '__main__':
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('--focus', '-f', dest='foc', type='float', default=6.25)
    opts, pos = parser.parse_args()
    test_case(opts.foc)


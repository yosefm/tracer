import numpy as N
import math
import pylab as P

from tracer.tracer_engine import TracerEngine
from tracer.ray_bundle import RayBundle, solar_disk_bundle
from tracer.spatial_geometry import rotx, generate_transform

from tracer.boundary_shape import BoundarySphere
from tracer.assembly import Assembly
from tracer.object import AssembledObject
from tracer.surface import Surface

from tracer.models.one_sided_mirror import one_sided_receiver
from tracer.paraboloid import ParabolicDishGM
import tracer.optics_callables as opt

def plot_hits(x, y, energy, bins=100):
    """
    Generates a 2D histogram of hits by their x,y coordinates, weighted by the
    energy absorbed in each hit. Plots the histogram using Pylab.
    
    Returns:
    f - a pylab figure object with the histogram on it, not shown yet.
    """
    H, xbins, ybins = N.histogram2d(x, y, bins, weights=energy)
    f = P.figure()
    P.imshow(H.T)
    P.colorbar()
    return f
    
def test_case(focus):
    # Case parameters (to be moved out:
    w = 10.
    h = 10.
    D = 5.
    
    num_rays = 10000
    center = N.c_[[0, 2., 2.]]
    x = -1/(math.sqrt(2))
    direction = N.array([0,x,x])
    radius_sun = 2.5 
    ang_range = 0.0005
    iterate = 2
    min_energy = 0.05
    
    # Model:
    receiver, rec_obj = one_sided_receiver(w, h)
    receiver_frame = generate_transform(N.r_[1.,0,0], 3*N.pi/4., N.c_[[0, 2.5, 2.5]])
    rec_obj.set_transform(receiver_frame)
    
    dish_surf = Surface(ParabolicDishGM(D, focus), opt.perfect_mirror)
    dish = AssembledObject(surfs=[dish_surf])
    dish.set_transform(rotx(-N.pi/4.))
    
    assembly = Assembly(objects=[rec_obj, dish])
    
    # Rays:
    sun = solar_disk_bundle(num_rays, center, direction, radius_sun, ang_range)
    sun.set_energy(N.ones(num_rays))
    sun.set_ref_index(N.ones(num_rays))
    
    # Do the tracing:
    engine = TracerEngine(assembly)
    engine.ray_tracer(sun, iterate, min_energy)
    
    # Plot:
    energy, pts = receiver.get_optics_manager().get_all_hits()
    x, y = receiver.get_geometry_manager().global_to_local(pts)[:2]
    f = plot_hits(x, y, energy)
    f.show()

if __name__ == '__main__':
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('--focus', '-f', dest='foc', type='float', default=6.25)
    opts, pos = parser.parse_args()
    test_case(opts.foc)
    P.show()

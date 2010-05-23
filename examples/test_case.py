import numpy as N
import math
import pylab as P

from tracer.tracer_engine import TracerEngine
from tracer.sources import solar_disk_bundle
from tracer.spatial_geometry import rotx

from tracer.models.tau_minidish import MiniDish

def plot_hits(hist):
    """
    Generates a PyLab plot of the histogrammed hits from a receiver
    
    Returns:
    f - a pylab figure object with the histogram on it, not shown yet.
    """
    f = P.figure()
    P.imshow(hist.T)
    P.colorbar()
    return f
    
def test_case(focus, num_rays=100):
    # Case parameters (to be moved out:
    D = 5.
    side = 0.4
    h_depth = 0.7
    
    center = N.c_[[0, 7., 7.]]
    x = -1/(math.sqrt(2))
    direction = N.array([0,x,x])
    radius_sun = 2.5 
    ang_range = 0.0005
    
    iterate = 100
    min_energy = 0.05
    
    # Model:
    assembly = MiniDish(D, focus, 0.9, focus + h_depth, side, h_depth, 0.9)
    assembly.set_transform(rotx(-N.pi/4))
    
    # Rays:
    sun = solar_disk_bundle(num_rays, center, direction, radius_sun, ang_range)
    sun.set_energy(N.ones(num_rays))
    sun.set_ref_index(N.ones(num_rays))
    
    # Do the tracing:
    engine = TracerEngine(assembly)
    engine.ray_tracer(sun, iterate, min_energy)
    
    # Plot:
    f = plot_hits(assembly.histogram_hits()[0])
    f.show()

if __name__ == '__main__':
    import optparse
    
    usage="""
    Create a dish with a square homogenizer, rotated by quarter-circle, and 
    shows the distribution of energy on the receiver.
    """
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--focus', '-f', dest='foc', type='float', default=6.25,
        help="Dish focal length, default: %default")
    parser.add_option('--num-rays', '-n', dest='num_rays', type='int', default=100,
        help="Number of rays in the initial bundle, default %default")
    opts, pos = parser.parse_args()
    test_case(opts.foc, opts.num_rays)
    P.show()

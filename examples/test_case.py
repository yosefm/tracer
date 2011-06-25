import numpy as N
import math
import pylab as P

from tracer.tracer_engine import TracerEngine
from tracer.sources import solar_disk_bundle
from tracer.spatial_geometry import rotx

from tracer.models.tau_minidish import MiniDish

def plot_hits(hist, extent):
    """
    Generates a PyLab plot of the histogrammed hits from a receiver
    
    Arguments:
    hist - a 2D array with the 2D histogram to plot.
    extent - the (left, right, bottom, top) of the histogram axes.
    
    Returns:
    f - a pylab figure object with the histogram on it, not shown yet.
    """
    f = P.figure()
    P.imshow(hist.T, extent=extent, interpolation='nearest')
    P.colorbar()
    return f
    
def test_case(focus, num_rays=100, h_depth=0.7, side=0.4):
    # Case parameters (to be moved out:
    D = 5.
    
    center = N.c_[[0, 7., 7.]]
    x = -1/(math.sqrt(2))
    direction = N.array([0,x,x])
    radius_sun = 2.5 
    ang_range = 0.005
    
    iterate = 100
    min_energy = 1e-6
    
    # Model:
    assembly = MiniDish(D, focus, 0.9, focus + h_depth, side, h_depth, 0.9)
    assembly.set_transform(rotx(-N.pi/4))
    
    # Rays:
    sun = solar_disk_bundle(num_rays, center, direction, radius_sun, ang_range,
        flux=1000.)
    
    # Do the tracing:
    engine = TracerEngine(assembly)
    engine.ray_tracer(sun, iterate, min_energy)
    
    # Plot, scale in suns:
    f = plot_hits(assembly.histogram_hits()[0]/(side/50)**2/1000., (-side/2., side/2., -side/2., side/2.))
    f.show()

if __name__ == '__main__':
    import optparse
    
    usage="""
    Create a 5-m diameter dish with a square homogenizer, rotated by quarter-circle,
    and shows the distribution of energy on the receiver.
    """
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--focus', '-f', dest='foc', type='float', default=6.25,
        help="Dish focal length, default: %default")
    parser.add_option('--num-rays', '-n', dest='num_rays', type='int', default=100,
        help="Number of rays in the initial bundle, default %default")
    parser.add_option('--hdepth', dest='hdepth', type='float', default=0.7,
        help="Homogenizer depth, default %default")
    parser.add_option('--receiver-side', '-s', dest='side', type='float', default=0.4,
        help="Side of square receiver area, default %default")
    opts, pos = parser.parse_args()
    
    test_case(opts.foc, opts.num_rays, opts.hdepth, opts.side)
    P.show()

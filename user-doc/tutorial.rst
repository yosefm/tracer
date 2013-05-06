
The Tracer Tutorial
===================

The basic use of Tracer is simple: create an assembly representing your optical
system; create a ray bundle to trace through the assembly; and feed these to
the tracer engine. Results, whatever they may be are obtained from the tracer
engine's tree of rays, as well as results accumulated in participating
objects.

Here we will follow an example that creates a parabolic dish, fires a ray
bundle representing the solar light, and shows a map of the flux on the focal
plane. The full example code is in examples/test_case.py

Building Blocks
---------------
The first thing to do is to import the required parts of the tracer::

    from tracer.tracer_engine import TracerEngine
    from tracer.ray_bundle import solar_disk_bundle
    from tracer.spatial_geometry import rotx
    from tracer.models.tau_minidish import MiniDish

The dish assembly is already there to be imported. Tracer contains some basic
assemblies that model often-used optical systems. Another convenience is the
solar_disk_bundle function, which creates a pillbox-sunshape RayBundle
instance, which we can feed to a TracerEngine instance.

Note that tracer contains a spatial_geometry module, which deals with 
homogenous transformation matrices mostly.

We also need to import NumPy and math for setting various parameters, and Pylab
for the final output::

    import numpy as N
    import math
    import pylab as P


Setup
-------

Most of the work is done in a function called ``test_case``. It starts as follows::

    def test_case(focus, num_rays=100):

It receives the dish's focal length, and the number of rays to use. Now we
create the assembly::
    
    D = 5.
    side = 0.4
    h_depth = 0.7
    assembly = MiniDish(D, focus, 0.9, focus + h_depth, side, h_depth, 0.9)
    assembly.set_transform(rotx(-N.pi/4))

In the last line the entire dish assembly is rotated around the X axis. This 
affects the dish surface, as well as the homogenizer and receiver which are
part of a MiniDish assembly. The receiver is a square placed slightly beyond
the focal plane, and the homogenizer is a square of mirrors (a kaleidoscope)
that directs all rays to the receiver and is placed before the receiver.

To make the sun shine, we set the center, direction, radius and angular range
(half-angle of the light cone) of the bundle to be generated::
    
    center = N.c_[[0, 7., 7.]]
    x = -1/(math.sqrt(2))
    direction = N.array([0,x,x])
    radius_sun = 2.5
    ang_range = 0.005 # 5 milirad, approx. the sun's range.
    
    sun = solar_disk_bundle(num_rays, center, direction, radius_sun, ang_range)
    sun.set_energy(N.ones(num_rays))
    sun.set_ref_index(N.ones(num_rays))

The bundle is placed behind the dish's receiver, facing the dish. To trace the
rays through the dish::
    
    engine = TracerEngine(assembly)
    engine.ray_tracer(sun, iterate, min_energy)

Output
------
though it is possible to calculate the flux map from the rays impinging on the
receiver by reading the status of the tracer engine, we will not do so here.
Instead, the MiniDish class has its own method for this, ``histogram_hits``.
We use this method::

    assembly.histogram_hits()[0]

Which we then can pass to a function that uses Pylab to show the histogram::

    def plot_hits(hist):
        f = P.figure()
        P.imshow(hist.T)
        P.colorbar()
        return f

And show it the usual way.


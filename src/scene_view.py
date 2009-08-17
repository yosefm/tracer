from enthought.mayavi import mlab
from numpy import *

from flat_surface import FlatSurface
from assembly import Assembly
from object import AssembledObject
from ray_bundle import RayBundle
from tracer_engine import TracerEngine

import spatial_geometry as G

def show_rays(tree, escaping_len):
    """
    Given the tree data structure from the ray tracing engine, 
    3D-plot the rays.
    """
    # Add an empty level to the tree just to have a simpler loop:
    tree.append(RayBundle.empty_bund())
    
    for level in xrange(len(tree) - 1):
        start_rays = tree[level]
        end_rays = tree[level + 1]
        parents = end_rays.get_parent()
        
        for ray in xrange(start_rays.get_num_rays()):
            if ray in parents:
                # Has a hit on another surface
                first_child = where(ray == parents)[0][0]
                endpoints = c_[
                    start_rays.get_vertices()[:,ray],
                    end_rays.get_vertices()[:,first_child] ]
            else:
                # Escaping ray.
                endpoints = c_[
                    start_rays.get_vertices()[:,ray],
                    start_rays.get_vertices()[:,ray] + start_rays.get_directions()[:,ray]*escaping_len ]
            
            mlab.plot3d(*endpoints)
    tree.pop()

x, y = mgrid[-5:5, -5:5]
z = zeros_like(x)
surf1 = mlab.mesh(x, y, z, color=(1,0,0))

x, y = mgrid[-5:5, -5:5]
z = zeros_like(x)
surf2 = mlab.mesh(x, y/sqrt(2), y/sqrt(2), color=(0,0,1))

nrm = 1/(math.sqrt(2))
dir = c_[[0,-nrm, nrm],[0,0,-1]]
position = c_ [[0,2,1],[0,2,1]]

bund = RayBundle()
bund.set_vertices(position)
bund.set_directions(dir)
bund.set_ref_index(r_[[1,1,1]]) 

rot1 = G.general_axis_rotation([1,0,0], pi/4)
energy = array([1,1])
bund.set_energy(energy)

surf1 = FlatSurface(rotation=rot1, width=10,height=10)
surf2 = FlatSurface(width=10,height=10)
assembly = Assembly()
object = AssembledObject()
object.add_surface(surf1)
object.add_surface(surf2)
assembly.add_object(object)

engine = TracerEngine(assembly)
params = engine.ray_tracer(bund, 5, .05)[0]
show_rays(engine.tree, 1)

mlab.view(0, -90)
mlab.roll(0)
mlab.show()

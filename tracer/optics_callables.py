# A collection of callables and tools for creating them, that may be used for
# the optics-callable part of a Surface object.

import optics
import ray_bundle
import numpy as N

def gen_reflective(absorptivity):
    """
    Generates a function that represents the optics of an opaque, absorptive
    surface with specular reflections.
    
    Arguments:
    absorptivity - the amount of energy absorbed before reflection.
    
    Returns:
    refractive - a function with the signature required by Surface.
    """
    def refractive(geometry, rays, selector):
        outg = ray_bundle.RayBundle()
        outg.set_vertices(geometry.get_intersection_points_global(selector))
        outg.set_directions(optics.reflections(
            rays.get_directions()[:,selector],
            geometry.get_normals(selector)))
        outg.set_energy(rays.get_energy()[selector]*(1 - absorptivity))
        outg.set_parent(N.where(selector)[0]) # Each ray is reflected in order
        # Moving in the same medium, no change of ref_index
        outg.set_ref_index(rays.get_ref_index()[selector])
        return outg
    return refractive


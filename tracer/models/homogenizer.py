# Model a simple specular-reflective homogenizer using one-sided mirror.

import numpy as N

from ..assembly import Assembly
from ..object import AssembledObject
from .one_sided_mirror import rect_one_sided_mirror

from .. import spatial_geometry as sp

def rect_homogenizer(aperture_xdim, aperture_ydim, height, opt_eff):
    """
    Generate an assembly representing a rectangular homogenizer, composed of
    four one-sided mirrors.
    
    Arguments:
    aperure_xdim - the length of the aperture along the local x axis
    aperure_ydim - the length of the aperture along the local y axis
    height - the mirrors will stand such that they extand from z=0 to z=height
    opt_eff - the optical efficiency of each mirror (the complement of its 
        absorption)
    
    Returns:
    An Assembly instance, with four objects inside, one for each wall.
    """
    abs = 1 - opt_eff
    wall_xp = rect_one_sided_mirror(height, aperture_ydim, abs)
    wall_xp.set_transform(
        N.dot(sp.translate(aperture_xdim/2., 0, height/2.), sp.roty(-N.pi/2.)))
    
    wall_xn = rect_one_sided_mirror(height, aperture_ydim, abs)
    wall_xn.set_transform(
        N.dot(sp.translate(-aperture_xdim/2., 0, height/2.), sp.roty(N.pi/2.)))
    
    wall_yp = rect_one_sided_mirror(aperture_xdim, height, abs)
    wall_yp.set_transform(
        N.dot(sp.translate(0, aperture_ydim/2., height/2.), sp.rotx(N.pi/2.)))
    
    wall_yn = rect_one_sided_mirror(aperture_xdim, height, abs)
    wall_yn.set_transform(
        N.dot(sp.translate(0, -aperture_ydim/2., height/2.), sp.rotx(-N.pi/2.)))
    
    return Assembly(objects=[wall_xp, wall_xn, wall_yp, wall_yn])

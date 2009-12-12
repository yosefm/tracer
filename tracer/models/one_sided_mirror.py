# Implement an object with a front reflective surface and an opaque back.

from object import AssembledObject
from surface import Surface
from flat_surface import RectPlateGM
import optics_callables as opt

from numpy import r_

def rect_one_sided_mirror(width, height, absorptivity):
    """
    construct an object with two surfaces: one on the XY plane, that is
    specularly reflective, and one slightly below (negative z), that is opaque.
    
    Arguments:
    width - the extent along the x axis in the local frame.
    height - the extent along the y axis in the local frame.
    absorptivity - the ratio of energy incident on the reflective side that's
        not reflected back.
    """
    front = Surface(RectPlateGM(width, height), 
        opt.gen_reflective(absorptivity))
    back = Surface(RectPlateGM(width, height), opt.gen_reflective(1.),
        location=r_[0., 0., -1e-10])
    obj = AssembledObject(surfs=[front, back])
    return obj

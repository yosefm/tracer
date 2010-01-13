# An assembly modeling the PETAL dish, located at Sde Boker, Israel [1]
#
# References:
# [1] Biryukov, S. Determining the optical properties of PETAL, the 400 m(2)
#     parabolic dish at Sede Boqer, 2004, J. of Solar Energy Engineering.

from .. import optics_callables as opt
from ..surface import Surface
from ..paraboloid import HexagonalParabolicDishGM
from .homogenized_local_receiver import HomogenizedLocalReceiver

class PETAL(HomogenizedLocalReceiver):
    def __init__(self, diameter, focal_length, dish_opt_eff,\
        receiver_pos, receiver_side, homogenizer_depth, homog_opt_eff,\
        receiver_aspect=1.):
        """
        Arguments:
        diameter - of the circle bounding the hexagonal aperture. 
        focal_length - of the parabolic dish
        dish_opt_eff - the optical efficiency of the dish
        receiver_pos - the distance along the optical axis from the dish to the
            receiver's end surface - the PV panel (should be about the focal 
            length)
        receiver_side - the receiver is square, with this side length.
        homogenizer_depth - the homogenizer has base dimensions to fit the PV
            square, and this height.
        homog_opt_eff - the optical efficiency of each mirror in the homogenizer
        receiver_aspect - allows creation of a non-square homogenizer. If this
            is set to a number not 1, then the x dimension will be changed, y
            remains == receiver_side
        """
        dish_surf = Surface(HexagonalParabolicDishGM(diameter, focal_length), 
            opt.Reflective(1 - dish_opt_eff))
        receiver_dims = (receiver_side, receiver_side*receiver_aspect)
        HomogenizedLocalReceiver.__init__(self, dish_surf, receiver_pos, \
            receiver_dims, homogenizer_depth, homog_opt_eff)
        
        # for later interrogation:
        self._ext_dims = (diameter, receiver_pos)
        
    def get_external_dimensions(self):
        """
        Returns the external dimensions (the ones you would put in an assembly
        drawing, the bounding cylinder dimensions) of the entire assembly.
        
        Returns:
        diameter - of the dish
        full_height - from the dish base to the receiver surface.
        """
        return self._ext_dims

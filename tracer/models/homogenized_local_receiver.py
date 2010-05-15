# A base class for solar collectors with a main reflector and square
# homogenizer after the focus, with a receiver surface at the end of the
# homogenizer.

import numpy as N

from .. import spatial_geometry as sp
from ..assembly import Assembly
from ..object import AssembledObject

from .one_sided_mirror import one_sided_receiver
from .homogenizer import rect_homogenizer

class HomogenizedLocalReceiver(Assembly):
    def __init__(self, main_reflector, receiver_pos, receiver_dims, \
        homogenizer_depth, homog_opt_eff):
        """
        Arguments:
        main_reflector - a Surface object representing the reflector that
            focuses rays into a homogenized receiver.
        receiver_pos - the distance along the optical axis from the main reflector to the
            receiver's end surface - the PV panel (should be about the focal 
            length)
        receiver_dims - if scalar, the receiver is square, with this side
            length. If a tuple, it's the x, y lengths respectively.
        homogenizer_depth - the homogenizer has base dimensions to fit the PV
            square, and this height.
        homog_opt_eff - the optical efficiency of each mirror in the homogenizer
        """
        if type(receiver_dims) is type(tuple()):
            self._sides = receiver_dims
        else:
            self._sides = (receiver_dims, receiver_dims)
        self._rec_pos = receiver_pos
        
        self._rec, rec_obj = one_sided_receiver(*self._sides)
        receiver_frame = N.dot(sp.translate(0, 0, receiver_pos), sp.rotx(N.pi))
        rec_obj.set_transform(receiver_frame)
        
        self._hom = rect_homogenizer(self._sides[0], self._sides[1], \
            homogenizer_depth, homog_opt_eff)
        self._hom.set_transform(receiver_frame)
        
        self._mr = main_reflector
        refl = AssembledObject(surfs=[main_reflector])
        Assembly.__init__(self, objects=[rec_obj, refl], subassemblies=[self._hom])
    
    def get_receiver_surf(self):
        """for anyone wishing to directly access the receiver"""
        return self._rec
    
    def get_homogenizer(self):
        """Direct access to the homogenizer subassembly"""
        return self._hom
    
    def get_main_reflector(self):
        return self._mr
    
    def histogram_hits(self, bins=50):
        """
        Generates a 2D histogram of energy absorbed at the receiver surface,
        assuming a trace has been run using this assembly.
        
        Arguments:
        bins - hom many bins per axis to use (default 50)
        
        Returns:
        H - a 2D array with the energy falling on each bean, x axis along
            the first dimension, y along second
        xbins, ybins - the edges of the bins (so one point more than the number
            of beans for each axis)
        
        See Also:
        numpy.histogram2D()
        """
        energy, pts = self._rec.get_optics_manager().get_all_hits()
        x, y = self._rec.global_to_local(pts)[:2]
        rngx = self._sides[0]/2.
        rngy = self._sides[1]/2.
        
        H, xbins, ybins = N.histogram2d(x, y, bins, \
            range=([-rngx,rngx], [-rngy,rngy]), weights=energy)
        return H, xbins, ybins

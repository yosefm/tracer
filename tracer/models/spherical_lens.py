"""
Readymade spherical lens optical object.

References:
.. [1] Warren J. Smith, Modern Optical engineering, SPIE Press, 4th ed., 2008
"""

from ..object import AssembledObject
from ..surface import Surface
from ..flat_surface import RoundPlateGM
from ..sphere_surface import CutSphereGM
from ..boundary_shape import BoundaryPlane
from ..cylinder import FiniteCylinder
from ..optics_callables import RefractiveHomogenous as Refractive
from ..spatial_geometry import rotx

import numpy as N

class SphericalLens(AssembledObject):
    def __init__(self, diameter, depth, R1, R2, refr_idx, transform=None):
        """
        Create at least two lens surfaces (back and front), and possibly a
        cylindrical face to close the object if its diameter is smaller than
        the diameter at which the two surfaces intersect each other.
        
        Note: if the optical path is thought of as going from left to right,
        the the positive Z axis is 'left'. It is also called 'up' due to the
        usual terminology in solar systems.
        
        Arguments:
        diameter - of the lens aperture.
        depth - distance between back and front surfaces along the lens's
            optical axis.
        R1, R2 - the radii of the front and back surfaces, respectively (the 
            front is the surface facing rays coming down the Z axis). Positive
            radius indicates that the center of curvature is down the Z axis
            from the surface, and either 0, None or infinity indicate planar face.
        refr_idx - refractive index of the material of the lens.
        """
        flip_side = rotx(N.pi)[:3,:3]
        
        # Front surface:
        if R1 in [0, None, N.inf, -N.inf]:
            self._front = Surface(
                RoundPlateGM(diameter/2.),
                Refractive(1., refr_idx))
            R1 = B.inf
        else:
            z = N.sqrt(R1**2 - diameter**2/4.) # location of cut plane
            if R1 > 0:
                sect1 = BoundaryPlane(location=N.r_[0, 0, z])
            else:
                sect1 = BoundaryPlane(location=N.r_[0, 0, -z], rotation=flip_side)
            sphere = CutSphereGM(radius=abs(R1), bounding_volume=sect1)
            
            refr=Refractive(1., refr_idx)
            self._front = Surface(geometry=sphere, optics=refr)
        
        # Back surface:
        if R2 in [0, None, N.inf, -N.inf]:
            self._back = Surface(
                RoundPlateGM(diameter/2.),
                Refractive(1., refr_idx),
                rotation=flip_side)
            R2 = N.inf
        else:
            z = N.sqrt(R2**2 - diameter**2/4.) # location of cut plane
            if R2 > 0:
                sect2 = BoundaryPlane(location=N.r_[0, 0, z])
            else:
                sect2 = BoundaryPlane(location=N.r_[0, 0, -z], rotation=flip_side)
            sphere = CutSphereGM(radius=abs(R2), bounding_volume=sect2)

            refr=Refractive(1., refr_idx)
            self._back = Surface(geometry=sphere, optics=refr)
        
        # While locating the surfaces, we'll calculate bounding cylinder
        # dimensions, and create it after the fact.
        cyl_height = 0
        cyl_loc = 0
        
        # Locate the planes such that the second focal point is at Z = -f,
        # where f is the focal length found from the lensmaker equation.
        # see [1] p.46, eqn. 3.21a
        opt_power = (refr_idx - 1)*(1./R1 - 1./R2 + depth*(refr_idx - 1)/R1/R2/refr_idx)
        f = 1./opt_power
        
        # Back primary point's distance from back surface:
        pd = f*depth*(refr_idx - 1)/refr_idx/R1
        if R2 != N.inf:
            locb = pd - R2
            self._back.set_location(N.r_[0., 0., locb])
            
            cyl_loc += (locb + sect2.get_location()[2])/2.
            cyl_height -= locb + sect2.get_location()[2]
        
        # Front primary point's distance from front surface:
        if R1 != N.inf:
            locf = pd + depth - R1
            self._front.set_location(N.r_[0., 0., locf])
            
            cyl_loc += (locf + sect1.get_location()[2])/2.
            cyl_height += locf + sect1.get_location()[2]
        
        # Create the bounding cylinder:
        if cyl_height > 0:
            self._cyl = Surface(
                FiniteCylinder(diameter, cyl_height),
                Refractive(refr_idx, 1.), location=N.r_[0., 0., cyl_loc])
            surfs = [self._front, self._back, self._cyl]
        else:
            surfs=[self._front, self._back]
        
        # Finalize the object:
        AssembledObject.__init__(self, surfs=surfs, transform=transform)
        self._f = f
    
    def focal_length(self):
        """
        Reports the lens' calculated effective focal length - the distance from
        the back primary point (at Z=0) to the back focal point.
        """
        return self._f


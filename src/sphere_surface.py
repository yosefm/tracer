# Implements spherical surface 

from surface import UniformSurface
import optics
from ray_bundle import RayBundle
from boundary_shape import BoundarySphere
import numpy as N
from quadric import QuadricSurface
import pdb

class SphereSurface(QuadricSurface):
    """
    Implements the geometry of a spherical surface.  
    """
    def __init__(self, location=None, absorptivity=0., radius=1., mirror=True):
        """
        Arguments:
        location of center, rotation, absorptivity - passed along to the base class.
        boundary - boundary shape defining the surface
        Private attributes:
        _rad - radius of the sphere
        _loc - location of the center of the sphere
        _boundary - boundary shape defining the surface
        _temp_loc - holds the value of a temporarily transformed center, for use of 
        calculations by trace engine
        _transform - the transformation of the sphere surface into the frame of the parent
        object. Within it's own local coordinate system the sphere is assume to be centered
        about the origin
        _inner_n & _outer_n - describe the refractive indices on either side of the surface;
        note that nothing defines the inside or outside of a surface and it is arbitrarily
        assigned to the surface that is already facing the air
        _mirror - indicates if the surface is fully reflective
        """
        QuadricSurface.__init__(self, location, None, absorptivity, mirror)
        self.set_radius(radius)  

    def get_radius(self):
        return self._rad
    
    def set_radius(self, rad):
        if rad <= 0:
            raise ValuError("Radius must be positive")
        self._rad = rad
     
    def transform_frame(self, transform):
        self._temp_loc = N.dot(transform, self._loc)

    def get_normal(self, dot, hit, c):
        normal = ((hit - c) if dot <= 0 else  (c - hit))[:,None]
        normal = normal/N.linalg.norm(normal)
        return normal

    # Ray handling protocol:
    def get_ABC(self, ray_bundle):
        """
        Deals wih a ray bundle intersecting with a sphere
        Arguments:
        ray_bundle - the incoming bundle 
        Returns a 1D array with the parametric position of intersection along
        each ray.  Rays that miss the surface return +infinity
        """ 
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self._temp_loc[:3]
        
        # Solve the equations to find the intersection point:
        A = (d**2).sum(axis=0)
        B = 2*(d*(v - c[:,None])).sum(axis=0)
        C = ((v - c[:,None])**2).sum(axis=0) - self.get_radius()**2
        
        return A, B, C
    
    

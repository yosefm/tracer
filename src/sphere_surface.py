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
        """Finds the normal by taking the derivative and rotating it, returns the            
        information to the quadric class for calculations                                  
        Arguments:                                                                      
        dot - the dot product of the normal vector and the incoming ray, used to determine 
        which side is the outer surface (this is not relevant to the paraboloid since the  
        cross product determines it, but it is to the sphere surface)                     
        hit - the coordinates of an intersection                                            
        c - the center/vertex of the surface 
        """
        normal = ((hit - c) if dot <= 0 else  (c - hit))[:,None]
        normal = normal/N.linalg.norm(normal)
        return normal

    # Ray handling protocol:
    def get_ABC(self, ray_bundle):
        """ 
        Determines the variables forming the relevant quadric equation                         
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
    
    

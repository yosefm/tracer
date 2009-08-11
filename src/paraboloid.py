# Implements a circular paraboloid surface
import numpy as N
from quadric import QuadricSurface
import pdb
class Paraboloid(QuadricSurface):
    """Implements the geometry of a circular paraboloid surface"""
    def __init__(self, location=None, rotation=None, absorptivity=0., a=1., b=1., mirror=True):
        """               
        Arguments:                                                                           
        location of center, rotation, absorptivity - passed along to the base class.    
        boundary - boundary shape defining the surface                                    
        Private attributes:                                                                  
        self.a, self.b - describe the paraboloid 
        """ 
        QuadricSurface.__init__(self, location, rotation,  absorptivity, mirror)
        self.a = 1/(a**2)
        self.b = 1/(b**2)

    def transform_frame(self, transform):
        self.transform = transform

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
        partial_x = 2*hit[0]*self.a
        partial_y = 2*hit[1]*self.b
        normal = N.dot(self.transform[:3][:,:3], N.cross(N.array([1,0,partial_x]), N.array([0,1,partial_y]))[:,None])
        normal = normal/N.linalg.norm(normal)
        return normal  
    
    def get_ABC(self, ray_bundle):
        """
        Determines the variables forming the relevant quadric equation
        """

        d = N.dot(self.transform[:3][:,:3].T, ray_bundle.get_directions())
        v = N.empty((4,N.shape(d)[1]))
        for ray in xrange(ray_bundle.get_num_rays()):
            v[:,ray] = N.dot(N.linalg.inv(self.transform), N.vstack((ray_bundle.get_vertices()[:,ray][:,None],N.array([1])))).T
        v = v[:3]

        A = self.a*d[0]**2 + self.b*d[1]**2
        B = 2*self.a*d[0]*v[0] + 2*self.b*d[1]*v[1] - d[2] 
        C = self.a*v[0]**2 + self.b*v[1]**2 - v[2]
        
        return A, B, C
    

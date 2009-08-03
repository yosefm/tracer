# Implements a circular paraboloid surface
import numpy as N
from quadric import QuadricSurface
import pdb
class Paraboloid(QuadricSurface):
    """Implements the geometry of a circular paraboloid surface"""
    def __init__(self, location=None, rotation=None, absorptivity=0., a=1., b=1., mirror=True):
        """
        """ 
        QuadricSurface.__init__(self, location, rotation,  absorptivity, mirror)
        self.a = 1/(a**2)
        self.b = 1/(b**2)

    def transform_frame(self, transform):
        self.transform = transform

    def get_normal(self, dot, hit, c):
        # Find the normal by taking the derivative and rotating it
        partial_x = 2*hit[0]*self.a
        partial_y = 2*hit[1]*self.b
        normal = N.cross(N.array([1,0,partial_x]), N.array([0,1,partial_y]))[:,None]
        normal = normal/N.linalg.norm(normal)
        return normal  
    
    
    def get_ABC(self, ray_bundle):
        d = N.dot(self.transform, N.vstack((ray_bundle.get_directions(), N.array([1]))))     
        v = ray_bundle.get_vertices() 
        A = self.a*d[0]**2 + self.b*d[1]**2
        B = 2*self.a*d[0]*v[0] + 2*self.b*d[1]*v[1] + d[2] 
        C = self.a*v[0] + self.b*v[1] + v[2]
        
        
        return A, B, C
    

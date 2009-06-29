# Implements a sphere as a bounding shape to a surface to define the exact shape of the surface

import numpy as N

class BoundarySphere():
    def __init__(self, center,  radius):
        """Arguments:
        center - a 1D array
        radius - a float
        """
        self.center = center
        self.radius = radius
            
    def in_bounds(self, bund_vertices):
        """
        Returns a boolean array for whether or not a ray intersection was within the 
        bounding sphere
        Arguments: bund_vertices - an array of the vertices
        """
        selector = []
        for ray in range(N.shape(bund_vertices)[1]):
            selector.append((self.radius >= N.linalg.norm(bund_vertices[:,ray] - self.center)))
           
        return N.array(selector)

 

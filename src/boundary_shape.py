# Implements a sphere as a bounding shape to a surface to define the exact shape of the surface

import numpy as N

class BoundarySphere():

    def __init__(self, location=None,  radius=1.):
        """Arguments:
        location - a 1D array
        radius - a float
        """
        if location == None:
            location = N.zeros(3)
        self._loc = location
        self._temp_loc = self._loc
        self._radius = radius
            
    def in_bounds(self, bund_vertices):
        """
        Returns a boolean array for whether or not a ray intersection was within the 
        bounding sphere
        Arguments: bund_vertices - an array of the vertices
        """
        selector = []
        for ray in range(N.shape(bund_vertices)[0]):
            selector.append((self._radius >= N.linalg.norm(bund_vertices[ray] - self._temp_loc[:3])))
           
        return N.array(selector)

    def transform_frame(self, transform):
        """ 
        Transforms the center of the boundary shape
        """
        self._temp_loc = N.dot(transform, N.append(self._loc, N.c_[[1]]))
    

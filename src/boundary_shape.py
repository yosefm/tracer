# Implements a sphere as a bounding shape to a surface to define the exact shape of the surface

import numpy as N

class BoundarySphere():

    def __init__(self, center,  radius):
        """Arguments:
        center - a 1D array
        radius - a float
        """
        self._center = center
        self._temp_center = self._center
        self._radius = radius
            
    def in_bounds(self, bund_vertices):
        """
        Returns a boolean array for whether or not a ray intersection was within the 
        bounding sphere
        Arguments: bund_vertices - an array of the vertices
        """
        selector = []
        for ray in range(N.shape(bund_vertices)[1]):  
            selector.append((self._radius >= N.linalg.norm(bund_vertices[:,ray] - self._temp_center[:3])))
           
        return N.array(selector)

    def transform_frame(self, transform):
        """ 
        Transforms the center of the boundary shape
        """
        self._temp_center = N.dot(transform, N.append(self._center, N.c_[[1]]))
    

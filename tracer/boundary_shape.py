# Implements a sphere as a bounding shape to a surface to define the exact shape of the surface

import numpy as N
from has_frame import HasFrame

class BoundaryShape(HasFrame):
    """
    Represent a surface that encloses a volume, so that it is possible to check
    whether certain points are within that volume.
    """
    def __init__(self, location=None, rotation=None):
        """
        Arguments:
        location - a 1D array of 3 components, representing the 3D location of 
            the surface's frame's origin, in the containing object's frame.
        rotation - a 3x3 array representing the rotation matrix of the surface's 
        """
        HasFrame.__init__(self, location, rotation)
    
    def in_bounds(self, points):
        """
        Every subclass must implement a function that says which points are in 
        the enclosed volume.
        
        Arguments: 
        points - a 3 by n array for n 3D points
        
        Returns: 
        an n-length 1D boolean array stating for each point whether it is in 
        the bounds or not.
        """
        raise TypeError("Virtual function in_bounds() called. Implement " + \
            "this in a derived class")

class BoundarySphere(BoundaryShape):
    def __init__(self, location=None,  radius=1.):
        """
        Arguments:
        radius - a float
        
        Attributes:
        _radius - radius of the bounding sphere
        """
        BoundaryShape.__init__(self, location, None)
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
        Transforms the center of the boundary shape into the global coordinates; this occurs
        when the assembly or object containing the surface is transformed
        """
        self._temp_loc = N.dot(transform, N.append(self._loc, N.c_[[1]]))
    

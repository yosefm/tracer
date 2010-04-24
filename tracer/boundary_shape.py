# Implements a sphere as a bounding shape to a surface to define the exact shape of the surface

import numpy as N
from has_frame import HasFrame

class BoundaryShape(HasFrame):
    """
    Represent a surface that encloses a volume, so that it is possible to check
    whether certain points are within that volume.
    
    Boundary shapes also contain helper functions for automatically generating
    meshes for objects: boundary_recr_for_plane() helps the surfaces know the 
    extent of the required mesh.
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
        points - an n by 3 array for n 3D points
        
        Returns: 
        an n-length 1D boolean array stating for each point whether it is in 
        the bounds or not.
        """
        raise TypeError("Virtual function in_bounds() called. Implement " + \
            "this in a derived class")

    def bounding_rect_for_plane(self, transform):
        """
        Find a rectangle on the xy plane of a given frame, which contains the
        intersection of the boundary shape and the plane.
        
        Arguments: 
        transform - a 4x4 array, the homog. transf. matrix from the global
            coordinates to the frame whose xy plane intersects the boundary 
            shape.
        
        Returns:
        xmin, xmax, ynin, ymax - of the rect, in the xy plane of the frame,
        """
        raise TypeError("Virtual bounding_rect_for_plane() called. " + \
            "Implement this in a derived class")

class BoundarySphere(BoundaryShape):
    def __init__(self, location=None,  radius=1.):
        """
        Arguments:
        radius - a float
        location - a 3-component column vector representing the location of the
            sphere's center, passed to the base class.
        
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
        Arguments: bund_vertices - an array of the vertices (see base class)
        """
        return self._radius**2 >= ((bund_vertices - self._temp_loc[:3])**2).sum(axis=1)

    def transform_frame(self, transform):
        """ 
        Transforms the center of the boundary shape into the global coordinates; this occurs
        when the assembly or object containing the surface is transformed
        """
        self._temp_loc = N.dot(transform, N.append(self._loc, N.c_[[1]]))
    
    def bounding_rect_for_plane(self, transform):
        """
        Find a rectangle on the xy plane of a given frame, which contains the
        intersection of the boundary shape and the plane.

        Arguments:
        transform - a 4x4 array, the homog. transf. matrix from the global
            coordinates to the frame whose xy plane intersects the boundary
            shape.

        Returns:
        xmin, xmax, ynin, ymax - of the rect, in the xy plane of the frame,
        """
        cent_proj = N.dot(N.linalg.inv(transform), N.append(self._temp_loc, 1))
        Reff = N.sqrt(self._radius**2 - (self._temp_loc[2] - cent_proj[2])**2)
        return cent_proj[0] - Reff, cent_proj[0] + Reff, \
            cent_proj[1] - Reff, cent_proj[1] + Reff

class BoundaryCylinder(BoundaryShape):
    def __init__(self, diameter=1., location=None, rotation=None):
        """
        Defines an infinite cylinder along the Z axis as a volume in which
        intersection points are valid.
        """
        self._R = diameter/2.
        BoundaryShape.__init__(self, location, None)
    
    def in_bounds(self, vertices):
        """
        Returns a boolean array for whether or not a ray intersection was within the 
        bounding sphere.
        
        Arguments: 
        vertices - an array of the points to check for inclusion, (n,3)
        """
        local_xy = N.dot(self._temp_frame[:2], 
            N.vstack((vertices.T, N.ones(vertices.shape[1]))))
        return N.sum(local_xy**2, axis=0) <= self._R**2


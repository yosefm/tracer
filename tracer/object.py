# Defines an object class, where an object is defined as an assembly of surfaces.

import numpy as N
from spatial_geometry import general_axis_rotation
from assembly import Assembly

class AssembledObject(Assembly):
    """ Defines an assembly of surfaces as an object. The object has its own set of 
    coordinates such that each surface composing the object can be described in terms of
    the object's coordinate system, and thus the user can rotate or translate the entire
    object together as one piece.
    The object also tracks refractive indices as a ray bundle leaves or enters a new
    material.
    """
    def __init__(self, surfs=None, bounds=None, transform=None):
        """
        Attributes:
        surfaces - a list of Surface objects
        boundaries - a list of Boundary objects that the surfaces are limited 
            by.
        transform - a 4x4 array representing the homogenous transformation 
            matrix of this object relative to the coordinate system of its 
            container
        """
        # Use the supplied values or some defaults:
        if surfs is None:
            self.surfaces = []
        else:
            self.surfaces = surfs
        
        if bounds is None:
            self.boundaries = []
        else:
            self.boundaries = bounds
        
        if transform is None:
            self.transform = N.eye(4)
        else:
            self.transform = transform
    
    def get_surfaces(self):
        return self.surfaces

    def add_surface(self, surface):
        """Adds a surface to the object
        Arguments:  surface - a surface object
        """
        self.surfaces.append(surface)
        surface.set_parent_object(self)

    def add_boundary(self, boundary):
        """Adds a boundary to the object. Surfaces not enclosed by the boundary
        sphere will not count as hit.
        Arguments: boundary - a spherical boundary objects
        """
        self.boundaries.append(boundary)

    def get_boundaries(self):
        return self.boundaries
    
    def set_transform(self, trans):
        """
        Sets the object's transformation relative to its assembly.
        Arguments:
        trans - a 4x4 array which is the homogenous transf. matrix from the 
            assembly frame to the object frame.
        """
        self.transform = trans
    
    def transform_object(self, assembly_transform):
        """Transforms an object if the assembly is transformed""" 
        for surface in self.surfaces:
            surface.transform_frame(N.dot(self.transform, assembly_transform))
        for boundary in self.boundaries:
            boundary.transform_frame(N.dot(self.transform, assembly_transform))

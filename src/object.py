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
    def __init__(self):
        """
        """ 
        self.surfaces = []

    def get_surfaces(self):
        return self.surfaces

    def add_surface(self, surface):
        """Adds a surface to the object
        Arguments:  surface - a surface object
        """
        self.surfaces.append(surface)

    def transform_object(self, assembly_transform):
        """Transforms an object if the assembly is transformed""" 
        for surface in xrange(len(self.surfaces)):
            self.surfaces[surface].transform_frame(N.dot(self.transform, assembly_transform))


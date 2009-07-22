# Define some basic surfaces for use with the ray tracer. From this minimal 
# hierarchy other surfaces should be derived to implement actual geometric
# operations.
#
# References:
# [1] John J. Craig, Introduction to Robotics, 3rd ed., 2005. 

# All surfaces are grey.

import numpy as N

class Surface(object):
    """Defines the base of surfaces that interact with rays.
    Each surface has a location vector and a rotation matrix. Together they define
    the frame of reference of the object.
    A surface may construct its own local coordinates from the vectors of the 
    rotation matrix.
    The rotation matrix is from the global to the local, so its columns are the 
    basis of the local coordinates, written in the global coordinates. See [1] p. 25
    """
    def __init__(self,  location=None,  rotation=None):
        # default location and rotation:
        if location is None:
            location = N.zeros(3)
        if rotation is None:
            rotation = N.eye(3)
        
        self.set_location(location)
        self.set_rotation(rotation)
    
    def get_location(self):
        return self._loc
    
    def get_rotation(self):
        return self._rot
    
    def set_location(self,  location):
        if location.shape != (3, ):
            raise ValueError("location must be a 1D 3-component array")
        self._loc = location
    
    def set_rotation(self,  rotation):
        if  rotation.shape != (3, 3):
            raise ValueError("rotation must be a 3x3 array")
        self._rot = rotation

    def set_parent_object(self, object):
        self.parent_object = object

class UniformSurface(Surface):
    """Implements an abstract surface whose material properties are independent of
    location.
    Currently only absorptivity is tracked.
    """
    def __init__(self,  location=None,  rotation=None,  absorptivity=0.):
        Surface.__init__(self,  location,  rotation)
        self.set_absorptivity(absorptivity)
    
    def get_absorptivity(self):
        return self._absorpt
    
    def set_absorptivity(self,  absorptivity):
        if  not 0 <= absorptivity <= 1:
            raise ValueError("Absorptivity must be in [0,1]")
        self._absorpt = absorptivity

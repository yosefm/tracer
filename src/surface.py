# Define some basic surfaces for use with the ray tracer. From this minimal 
# hierarchy other surfaces should be derived to implement actual geometric
# operations.
#
# References:
# [1] John J. Craig, Introduction to Robotics, 3rd ed., 2005. 

# All surfaces are grey.

import numpy as N
import pdb

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

    def set_inner_n(self, n):
        self._inner_n = n

    def set_outer_n(self, n):
        self._outer_n = n

    def get_inner_n(self):
        return self._inner_n

    def get_outer_n(self):
        return self._outer_n
    
    def set_transform(self, transform):
        self._transform = transform

    def get_transform(self):
        return self._transform

    def transform_frame(self, transform):
        self._temp_frame = N.dot(transform, self._transform)

    def get_ref_index(self, n, bund, selector):
        """                                                                                  
        Determines which refractive index to use based on the refractive index               
        the ray is currently travelling through                                              
        Also sets the new values for the ray bundle                                           
        Arguments: n - an array of the refractive indices of the materials each of             
        the rays in a ray bundle                                                              
        """
        new_n = n.copy()
        for ray in xrange(N.shape(n)[0]):
            if n[ray] == self.get_inner_n(): new_n[ray] = self.get_outer_n() 
            else: new_n[ray] = self.get_inner_n()
        # Prepares a new set of refractive indices for the next ray bundle. It includes
        # the new ref indices when a ray enters a material, and the same ref indices
        # if the ray is reflected. The new set of ref indices are not used until the
        # first iteration is done
        bund.set_temp_ref_index(N.hstack((new_n[selector], n[selector])))
        return new_n   
    
class UniformSurface(Surface):
    """Implements an abstract surface whose material properties are independent of
    location.
    Currently only absorptivity is tracked.
    """
    def __init__(self,  location=None,  rotation=None,  absorptivity=0., mirror=True):
        Surface.__init__(self,  location,  rotation)
        self.set_absorptivity(absorptivity)
        self.mirror = mirror
    
    def get_absorptivity(self):
        return self._absorpt
    
    def set_absorptivity(self,  absorptivity):
        if  not 0 <= absorptivity <= 1:
            raise ValueError("Absorptivity must be in [0,1]")
        self._absorpt = absorptivity

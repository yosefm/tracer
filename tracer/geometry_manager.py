# -*- coding: utf-8 -*-
# Define a base class for managing geometry of an optical surface.
# This is an abstract class defining the interface.
# TODO: explore abstract meta-classes in Python.

import numpy as N

class GeometryManager(object):
    def find_intersections(self, frame, ray_bundle):
        self._working_frame = frame
        self._working_bundle = ray_bundle
        
        # This must be extended to return the correct result!
        if type(self) is GeometryManager:
            raise TypeError("Find intersections must be extended by a base class")
    
    def done(self):
        """
        Discard internal data structures. After calling done(), the information
        on the latest trace iteration is lost. 
        """
        if hasattr(self, '_working_frame'):
            del self._working_frame
            del self._working_bundle
    
    def select_rays(self, idxs):
        pass
    
    def get_normals(self):
        pass
    
    def get_intersection_points_global(self):
        pass

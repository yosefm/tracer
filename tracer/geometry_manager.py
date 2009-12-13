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
    
    def get_normals(self, selector):
        pass
    
    def get_intersection_points_global(self, selector):
        pass
    
    def global_to_local(self, points):
        """
        Transform a set of points in the global coordinates back into the frame
        used during tracing.
        
        Arguments:
        points - a 3 x n array for n 3D points
        
        returns:
        local - a 3 x n array with the respective points in local coordinates.
        """
        return N.dot(N.linalg.inv(self._working_frame), 
            N.vstack((points, N.ones(points.shape[1]))))

# Define a base class for managing geometry of an optical surface.
# This is an abstract class defining the interface.
# TODO: explore abstract meta-classes in Python.

class GeometryManager(object):
    def find_intersections(self, frame, ray_bundle):
        self._working_frame = frame
        self._working_bundle = ray_bundle
        
        # This must be extended to return the correct result!
        if is_instance(self, GeometryManager):
            raise TypeError("Find intersections must be extended by a base class")
    
    def get_normals(self, selector):
        pass
    
    def get_intersection_points_global(self, selector):
        pass


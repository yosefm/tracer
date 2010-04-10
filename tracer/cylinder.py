"""
Geometry managers based on a cylinder along the Z axis.

References
----------

.. [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
"""

from .quadric import QuadricGM

class OpenCylinder(QuadricGM):
    """
    A cylindrical surface infinitely long on the Z axis.
    """
    def __init__(self, diameter):
        self._R = diameter/2.
        QuadricGM.__init__(self)
    
    def get_ABC(bundle):
        """
        Finds the coefficients of the quadratic equation for the intersection
        of a ray and the cylinder. See [1]_.
        
        Arguments:
        bundle - a RayBundle instance with rays for which to get the
            coefficients.
        
        Returns:
        A, B, C - satisfying A*t**2 + B*t + C = 0 such that the intersection
            points are at distance t from each ray's vertex.
        """
        # Transform the the direction and position of the rays temporarily into the
        # frame of the paraboloid for calculations
        d = N.dot(self._working_frame[:3,:3].T, ray_bundle.get_directions())
        v = N.dot(N.linalg.inv(self._working_frame),
            N.vstack((ray_bundle.get_vertices(), N.ones(d.shape[1]))))[:3]
        
        A = N.sum(d[:2]**2, axis=0)
        B = 2*N.sum(d[:2]*v[:2], axis=0)
        C = N.sum(v[:2]**2, axis=0) - self._R**2
        
        return A, B, C
    
    def _normals(verts, dirs):
        # Move to local coordinates
        hit = N.dot(N.linalg.inv(self._working_frame),
            N.vstack((verts.T, N.ones(hits.shape[0]))))
        dir_loc = N.dot(self._working_frame[:3,:3].T, dirs.T)
        
        # The local normal is made from the X,Y components of the vertex:
        local_norm = N.vstack((hit[:2], N.zeros(hit.shape[1])))
        local_norm /= N.sqrt(N.sun(hit[:2]**2, axis=0))
        
        # Choose whether the normal is inside or outside:
        local_norm[N.sum(local_norm[:2] * dir_loc[:2]) > 0] *= -1
        
        # Back to global coordinates:
        return N.dot(self._working_frame[:3,:3], local_norm)



# Implements a circular paraboloid surface
#
# References:
# [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
# [2] http://en.wikipedia.org/wiki/Parabola

import numpy as N
from quadric import QuadricGM

class Paraboloid(QuadricGM):
    """Implements the geometry of a circular paraboloid surface"""
    def __init__(self, a=1., b=1.):
        """               
        Arguments: 
        a, b - describe the paraboloid as z = (x/a)**2 + (y/b)**2
            (sorry, legacy)
        
        Private attributes:                                                                  
        a, b - describe the paraboloid as z = a*x**2 + b*y**2
        """ 
        QuadricGM.__init__(self)
        self.a = 1./(a**2)
        self.b = 1./(b**2)

    def _normals(self, hits, directs):
        """
        Finds the normal to the parabola in a bunch of intersection points, by
        taking the derivative and rotating it. Used internally by quadric.
        
        Arguments:
        hits - the coordinates of intersections, as an n by 3 array.
        directs - directions of the corresponding rays, n by 3 array.
        """
        hit = N.dot(N.linalg.inv(self._working_frame), 
            N.vstack((hits.T, N.ones(hits.shape[0]))))
        dir_loc = N.dot(self._working_frame[:3,:3].T, directs.T)
        partial_x = 2*hit[0]*self.a
        partial_y = 2*hit[1]*self.b
        
        local_normal = N.vstack((-partial_x, -partial_y, N.ones_like(partial_x)))
        local_unit = local_normal/N.sqrt(N.sum(local_normal**2, axis=0))
        down = N.sum(dir_loc * local_unit, axis=0) > 0
        local_unit[:,down] *= -1
        normals = N.dot(self._working_frame[:3,:3], local_unit)
        
        return normals  
    
    def get_ABC(self, ray_bundle):
        """
        Determines the variables forming the relevant quadric equation, [1]
        """
        # Transform the the direction and position of the rays temporarily into the
        # frame of the paraboloid for calculations
        d = N.dot(self._working_frame[:3,:3].T, ray_bundle.get_directions())
        v = N.dot(N.linalg.inv(self._working_frame), 
            N.vstack((ray_bundle.get_vertices(), N.ones(d.shape[1]))))[:3]
        
        A = self.a*d[0]**2 + self.b*d[1]**2
        B = 2*self.a*d[0]*v[0] + 2*self.b*d[1]*v[1] - d[2] 
        C = self.a*v[0]**2 + self.b*v[1]**2 - v[2]
        
        return A, B, C


import math
class ParabolicDishGM(Paraboloid):
    def __init__(self, diameter, focal_length):
        """
        A paraboloid that marks rays outside its diameter as missing. The
        parameters for the paraboloid's equation are determined from the focal
        length.
        
        Arguments:
        diameter - of the circular aperture created by cutting the paraboloid
            with a plane parallel to the xy plane.
        focal_length - distance of the focal point from the origin.
        """
        par_param = 2*math.sqrt(focal_length) # [2]
        Paraboloid.__init__(self, par_param, par_param)
        self._R = diameter/2.
    
    def find_intersections(self, frame, ray_bundle):
        """
        Extend the quadric surface's general routine by marking rays outside
        the diameter as missing.
        """
        ray_prm = Paraboloid.find_intersections(self, frame, ray_bundle)
        # Save a copy of the local coordinates of impact for use later
        local = N.dot(N.linalg.inv(self._working_frame), 
            N.vstack((self._vertices, N.ones(self._vertices.shape[1]))))
        
        # Use local coordinates to find distance on the local xy plane
        hit_dist = (local[:2]**2).sum(axis=0)
        ray_prm[hit_dist > self._R**2] = N.inf
        
        return ray_prm

class HexagonalParabolicDishGM(Paraboloid):
    def __init__(self, diameter, focal_length):
        """
        A paraboloid that marks rays outside a regular hexagon perimiter as
        missing. The parameters for the paraboloid's equation are determined
        from the focal length. The hexagon is oriented with two parallel to the
        Y axis.
        
        Arguments:
        diameter - of the circle bounding the regular hexagonal aperture of the
            dish.
        focal_length - distance of the focal point from the origin.
        """
        par_param = 2*math.sqrt(focal_length) # [2]
        Paraboloid.__init__(self, par_param, par_param)
        self._R = diameter/2.
    
    def find_intersections(self, frame, ray_bundle):
        """
        Extend the paraboloid's general routine by marking rays outside
        the hexagonal aperture as missing.
        """
        ray_prm = Paraboloid.find_intersections(self, frame, ray_bundle)
        # Save a copy of the local coordinates of impact for use later
        local = N.dot(N.linalg.inv(self._working_frame), 
            N.vstack((self._vertices, N.ones(self._vertices.shape[1]))))
        
        # Use local coordinates to find distance on the local xy plane
        abs_y = abs(local[1])
        abs_x = abs(local[0])
        outside = abs_x > math.sqrt(3)*self._R/2.
        outside |= abs_y > self._R - math.tan(N.pi/6.)*abs_x
        
        ray_prm[outside] = N.inf
        return ray_prm

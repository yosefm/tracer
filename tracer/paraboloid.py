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
        self._h = (diameter/2./par_param)**2
    
    def _select_coords(self, coords, prm):
        """
        Choose between two intersection points on a quadric surface.
        This implementation extends QuadricGM's behaviour by not choosing
        intersections outside the circular aperture.
        
        Arguments:
        coords - a 2 by 3 by n array whose each column is the global coordinates
            of one intersection point of a ray with the sphere.
        prm - the corresponding parametric location on the ray where the
            intersection occurs.

        Returns:
        The index of the selected intersection, or None if neither will do.
        """
        select = QuadricGM._select_coords(self, coords, prm) # defaults
        
        coords = N.concatenate((coords, N.ones((2,1,coords.shape[2]))), axis=1)
        local_z = N.sum(N.linalg.inv(self._working_frame)[None,2,:,None] * \
            coords, axis=1)
        under_cut = (local_z <= self._h) & (prm > 0)

        select[~N.logical_or(*under_cut)] = N.nan
        one_hit = N.logical_xor(*under_cut)
        select[one_hit] = N.nonzero(under_cut[:,one_hit])[0]

        return select

class HexagonalParabolicDishGM(Paraboloid):
    def __init__(self, diameter, focal_length):
        """
        A paraboloid that marks rays outside a regular hexagon perimeter as
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
    
    def _select_coords(self, coords, prm):
        """
        Choose between two intersection points on a quadric surface.
        This implementation extends QuadricGM's behaviour by not choosing
        intersections outside the hexagon aperture.
        
        Arguments:
        coords - a 2 by 3 by n array whose each column is the global coordinates
            of one intersection point of a ray with the sphere.
        prm - the corresponding parametric location on the ray where the
            intersection occurs.

        Returns:
        The index of the selected intersection, or None if neither will do.
        """
        select = QuadricGM._select_coords(self, coords, prm) # defaults

        coords = N.concatenate((coords, N.ones((2,1,coords.shape[2]))), axis=1)
        local = N.sum(N.linalg.inv(self._working_frame)[None,:2,:,None] * \
            coords[:,None,:,:], axis=2)
        
        abs_x = abs(local[:,0,:])
        abs_y = abs(local[:,1,:])
        outside = abs_x > math.sqrt(3)*self._R/2.
        outside |= abs_y > self._R - math.tan(N.pi/6.)*abs_x
        inside = (~outside) & (prm > 0)
        
        select[~N.logical_or(*inside)] = N.nan
        one_hit = N.logical_xor(*inside)
        select[one_hit] = N.nonzero(inside[:,one_hit])[0]

        return select


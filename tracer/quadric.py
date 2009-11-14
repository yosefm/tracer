# Implements spherical surface 
#
# References:
# [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm

import numpy as N
from geometry_manager import GeometryManager

class QuadricGM(GeometryManager):
    def find_intersections(self, frame, ray_bundle):
        """
        Register the working frame and ray bundle, calculate intersections
        and save the parametric locations of intersection on the surface.

        Arguments:
        frame - the current frame, represented as a homogenous transformation
            matrix stored in a 4x4 array.
        ray_bundle - a RayBundle object with the incoming rays' data.

        Returns:
        A 1D array with the parametric position of intersection along each of
            the rays. Rays that missed the surface return +infinity.
        """
        GeometryManager.find_intersections(self, frame, ray_bundle)
        
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self._working_frame[:3,3]
        
        params = []
        vertices = []
        norm = []
        
        # Gets the relevant A, B, C from whichever quadric surface, see [1]  
        A, B, C = self.get_ABC(ray_bundle)
        
        delta = B**2 - 4*A*C
    
        for ray in xrange(n):
            vertex = v[:,ray]

            if (delta[ray]) < 0:
                params.append(N.inf)
                vertices.append(N.empty([3,1]))
                norm.append(N.empty([3,1]))    
                continue
            
            if A[ray] <= 1e-10: 
                hit = -C[ray]/B[ray]
                hits = N.hstack((hit,hit))
            
            else: hits = (-B[ray] + N.r_[-1, 1]*N.sqrt(delta[ray]))/(2*A[ray])
            coords = vertex + d[:,ray]*hits[:,None]
            
            # Quadrics can have two intersections. Here we allow child classes
            # to choose based on own method:
            select = self._select_coords(coords, hits)
            
            # Returning None from _select_coords() means the ray missed anyway:
            if select is None:
                params.append(N.inf)
                vertices.append(N.empty([3,1]))
                norm.append(N.empty([3,1]))
                continue
            
            verts = N.c_[coords[select,:]]
            
            dot = N.vdot(c.T - coords[select,:], d[:,ray])
            normal = self.get_normal(dot, coords[select,:], c)
            
            params.append(hits[select])
            vertices.append(verts)
            norm.append(normal)
            
        # Storage for later reference:
        self._vertices = N.hstack(vertices)
        self._current_bundle = ray_bundle
        self._norm = N.hstack(norm)
        
        return N.array(params)
    
    def _select_coords(self, coords, prm):
        """
        Choose between two intersection points on a quadric surface.
        This is a default implementation that takes the first positive-
        parameter intersection point.
        
        The default behaviour is to take the first intersection not behind the
        ray's vertex (positive prm).
        
        Arguments:
        coords - a 2x3 array whose each row is the global coordinates of one
            intersection point of a ray with the sphere.
        prm - the corresponding parametric location on the ray where the 
            intersection occurs.
        
        Returns:
        The index of the selected intersection, or None if neither will do.
        """
        is_positive = N.where(prm > 0)[0]

        # If both are negative, it is a miss
        if len(is_positive) == 0:
            return None
        
        # If both are positive, us the smaller one
        if len(is_positive) == 2:
            return N.argmin(prm)
        else:
            # If either one is negative, use the positive one
            return is_positive[0]
        
    def get_normals(self, selector):
        """
        Report the normal to the surface at the hit point of selected rays in
        the working bundle.

        Arguments:
        selector - a boolean array stating which columns of the working bundle
            are active.
        """
        return self._norm[:,selector]
    
    def get_intersection_points_global(self, selector):
        """
        Get the ray/surface intersection points in the global coordinates.

        Arguments:
        selector - a boolean array stating which columns of the working bundle
            are active.

        Returns:
        A 3-by-n array for 3 spatial coordinates and n rays selected.
        """
        return self._vertices[:,selector]


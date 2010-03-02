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
        
        params = N.empty(n)
        params.fill(N.inf)
        vertices = N.empty((3,n))
        
        # Gets the relevant A, B, C from whichever quadric surface, see [1]  
        A, B, C = self.get_ABC(ray_bundle)
        delta = B**2 - 4*A*C
        
        # Get this outside the loop:
        pm = N.c_[[-1, 1]]
        any_inters = delta >= 0
        delta[any_inters] = N.sqrt(delta[any_inters])
        
        hits = N.empty((2, n))
        almost_planar = A <= 1e-10
        access_planar = any_inters & almost_planar
        access_quadric = any_inters & ~almost_planar
        hits[:,access_planar] = N.tile(-C[access_planar]/B[access_planar], (2,1))
        hits[:,access_quadric] = \
            (-B[access_quadric] + pm*delta[access_quadric])/(2*A[access_quadric])
        inters_coords = N.empty((2, 3, n))
        inters_coords[...,any_inters] = v[:,any_inters] + d[:,any_inters]*hits[:,any_inters].reshape(2,1,-1)
        
        for ray in N.where(any_inters)[0]:            
            # Quadrics can have two intersections. Here we allow child classes
            # to choose based on own method:
            select = self._select_coords(inters_coords[...,ray], hits[:,ray])
            # Returning None from _select_coords() means the ray missed anyway:
            if select is None: 
                any_inters[ray] = False
                continue
            
            params[ray] = hits[select, ray]
            vertices[:,ray] = inters_coords[select,:,ray]
            
        # Normals to the surface at the intersection points are calculated by
        # the subclass' _normals method.
        self._norm = N.empty((3,n))
        if any_inters.any():
            sides = N.sum((c - vertices[:,any_inters].T) * d[:,any_inters].T, axis=1)
            self._norm[:,any_inters] = self._normals(sides, vertices[:,any_inters].T, c)
        
        # Storage for later reference:
        self._vertices = vertices
        self._current_bundle = ray_bundle
        
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


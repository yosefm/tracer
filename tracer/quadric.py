# Implements quadric surfaces.
#
# References:
# [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm

import numpy as N
from geometry_manager import GeometryManager

class QuadricGM(GeometryManager):
    """
    A base class for quadric surfaces, to be derived for creation of specific
    quadric geometries. Each subclass should define the following methods:
    
    get_ABC(ray_bundle) - Given a RAyBundle instance, return A, B, C, the
        coefficients of a quadratic equation of t, the parametric position
        on each ray where it hits the surface (each of A, B, C is as long as
        the number of rays in ray_bundle).
    
    _normals(verts, dirs)
        Arguments:
        verts - an n by 3 array whose rows are points on the surace in global
            coordinates
        dirs - an n by 3 array whose columns are the respective incidence directions
        
        Returns:
        A 3 by n array with the rewpective normals to the surface at each of `verts`
    
    Additionally, overriding _select_coords(self, coords, prm) may be required.
    """
    
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
        
        any_inters = delta >= 0
        delta[any_inters] = N.sqrt(delta[any_inters])
        
        pm = N.c_[[-1, 1]]
        hits = N.empty((2, n))
        almost_planar = A <= 1e-10
        access_planar = any_inters & almost_planar
        access_quadric = any_inters & ~almost_planar
        hits[:,access_planar] = N.tile(-C[access_planar]/B[access_planar], (2,1))
        hits[:,access_quadric] = \
            (-B[access_quadric] + pm*delta[access_quadric])/(2*A[access_quadric])
        inters_coords = N.empty((2, 3, n))
        inters_coords[...,any_inters] = v[:,any_inters] + d[:,any_inters]*hits[:,any_inters].reshape(2,1,-1)
        
        # Quadrics can have two intersections. Here we allow child classes
        # to choose based on own method:
        select = self._select_coords(inters_coords, hits)
        missed_anyway = N.isnan(select)
        any_inters[missed_anyway] = False
        select = N.int_(select[any_inters])
        params[any_inters] = N.choose(select, hits[:,any_inters])
        vertices[:,any_inters] = N.choose(select, inters_coords[...,any_inters])
        
        # Normals to the surface at the intersection points are calculated by
        # the subclass' _normals method.
        self._norm = N.empty((3,n))
        if any_inters.any():
            self._norm[:,any_inters] = self._normals(vertices[:,any_inters].T,
                d[:,any_inters].T)
        
        # Storage for later reference:
        self._vertices = vertices
        
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
        is_positive = prm > 0
        select = N.empty(prm.shape[1])

        # If both are negative, it is a miss
        # This line also catches the cases of the last xor.
        select[N.logical_or(*is_positive)] = N.nan
        
        # If both are positive, use the smaller one
        select[N.logical_and(*is_positive)] = 0
        
        # If either one is negative, use the positive one
        one_pos = N.logical_xor(*is_positive)
        select[one_pos] = N.nonzero(is_positive[:,one_pos])[0]
        
        return select
        
    def select_rays(self, idxs):
        """
        With this method, the ray tracer informs the surface that of the
        registered rays, only those with the given indexes will be used next.
        This is used here to trim the internal data structures and save memory.
        
        Arguments:
        idx - an array of indexes referring to the rays registered in
            register_incoming()
        """
        self._idxs = idxs
        self._norm = self._norm[:,idxs].copy()
        self._vertices = self._vertices[:,idxs].copy()
    
    def get_normals(self):
        """
        Report the normal to the surface at the hit point of selected rays in
        the working bundle.
        """
        return self._norm
    
    def get_intersection_points_global(self):
        """
        Get the ray/surface intersection points in the global coordinates.

        Returns:
        A 3-by-n array for 3 spatial coordinates and n rays selected.
        """
        return self._vertices
    
    def done(self):
        """
        Discard internal data structures. This should be called after all
        information on the latest bundle's results have been extracted already.
        """
        del self._vertices
        del self._norm
        if hasattr(self, '_idxs'):
            del self._idxs
        GeometryManager.done(self)


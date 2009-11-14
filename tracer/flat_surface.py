# Implements a specularly reflecting, grey surface.

from numpy import linalg as LA
import numpy as N
from geometry_manager import GeometryManager

class FlatGeometryManager(GeometryManager):
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
        
        xy = frame[:3,:2]
        d = -ray_bundle.get_directions()
        v = ray_bundle.get_vertices() - frame[:3,3][:,None]
        n = ray_bundle.get_num_rays()
        
        # `params` holds the parametric location of intersections along x axis, 
        # y-axis and ray, in that order.
        params = N.empty((3, n))
        for ray in xrange(n):
            # Solve the linear equation system of the intersection point:
            eqns = N.hstack((xy,  d[:, ray][:, None]))
            if LA.det(eqns) != 0:
                params[:, ray] = N.dot(LA.inv(eqns), v[:, ray])
                continue
            # Singular matrix (parallel rays to the surface):
            params[:, ray].fill(N.inf)
        
        # Takes into account a negative depth
        # Note that only the 3rd row of params is relevant here!
        negative = params[2] < 0
        params[2, negative] = N.Inf
        
        # Storage for later reference:
        self._current_params = params[:2]
        return params[2]
    
    def get_normals(self, selector):
        """
        Report the normal to the surface at the hit point of selected rays in
        the working bundle.
        
        Arguments: 
        selector - a boolean array stating which columns of the working bundle
            are active.
        """
        return N.tile(self._working_frame[:3,2][:,None], (1, selector.sum()))
    
    def get_intersection_points_global(self, selector):
        """
        Get the ray/surface intersection points in the global coordinates.
        
        Arguments: 
        selector - a boolean array stating which columns of the working bundle
            are active.
        
        Returns:
        A 3-by-n array for 3 spatial coordinates and n rays selected.
        """
        return N.dot(self._working_frame[:3,:2],
            self._current_params[:,selector]) + self._working_frame[:3,3][:,None]


# -*- coding: utf-8 -*-
# Implements a specularly reflecting, grey surface.
# 
# Reference:
# [1]http://www.siggraph.org/education/materials/HyperGraph/raytrace/rayplane_intersection.htm

from numpy import linalg as LA
import numpy as N
from geometry_manager import GeometryManager

class FlatGeometryManager(GeometryManager):
    def find_intersections(self, frame, ray_bundle):
        """
        Register the working frame and ray bundle, calculate intersections
        and save the parametric locations of intersection on the surface.
        Algorithm taken from [1].
        
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
        v = ray_bundle.get_vertices() - frame[:3,3][:,None]
        n = ray_bundle.get_num_rays()
        
        # Vet out parallel rays:
        dt = N.dot(d.T, frame[:3,2])
        unparallel = abs(dt) > 1e-10
        
        # `params` holds the parametric location of intersections along x axis, 
        # y-axis and ray, in that order.
        params = N.empty((3, n))
        params.fill(N.inf)
        
        vt = N.dot(frame[:3,2], v[:,unparallel])
        params[2,unparallel] = -vt/dt[unparallel]
        
        # Takes into account a negative depth
        # Note that only the 3rd row of params is relevant here!
        negative = params[2] < 0
        params[2, negative] = N.Inf
        
        # Local coordinates on the surface:
        hitting = unparallel & ~negative
        hits = v[:,hitting] + params[2,hitting]*d[:,hitting]
        params[0,hitting] = N.dot(frame[:3,0], hits)
        params[1,hitting] = N.dot(frame[:3,1], hits)
        
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

class RectPlateGM(FlatGeometryManager):
    def __init__(self, width, height):
        """
        Arguments:
        width - the extent along the x axis in the local frame (sets self._w)
        height - the extent along the y axis in the local frame (sets self._h)
        """
        if width <= 0:
            raise ValueError("Width must be positive")
        if height <= 0:
            raise ValueError("Height must be positive")
        
        self._w = width
        self._h = height
        FlatGeometryManager.__init__(self)
        
    def find_intersections(self, frame, ray_bundle):
        """
        Extends the parent flat geometry manager by discarding in advance
        impact points outside a centered rectangle.
        """
        ray_prms = FlatGeometryManager.find_intersections(self, frame, ray_bundle)
        ray_prms[abs(self._current_params[0]) > self._w/2.] = N.inf
        ray_prms[abs(self._current_params[1]) > self._h/2.] = N.inf
        return ray_prms

class RoundPlateGM(FlatGeometryManager):
    def __init__(self, R):
        """
        Arguments:
        R - the plate's radius
        """
        if R <= 0:
            raise ValueError("Radius must be positive")
        
        self._R = R
        FlatGeometryManager.__init__(self)
    
    def find_intersections(self, frame, ray_bundle):
        """
        Extends the parent flat geometry manager by discarding in advance
        impact points outside a centered circle.
        """
        ray_prms = FlatGeometryManager.find_intersections(self, frame, ray_bundle)
        ray_prms[N.sum(self._current_params[:2]**2, axis=0) > self._R**2] = N.inf
        return ray_prms

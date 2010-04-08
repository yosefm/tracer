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
        
        # `params` holds the parametric location of intersections along the ray 
        params = N.empty(n)
        params.fill(N.inf)
        
        vt = N.dot(frame[:3,2], v[:,unparallel])
        params[unparallel] = -vt/dt[unparallel]
        
        # Takes into account a negative depth
        # Note that only the 3rd row of params is relevant here!
        negative = params < 0
        params[negative] = N.Inf
        
        self._params = params
        self._backside = dt > 0
        
        return params
        
    def select_rays(self, idxs):
        """
        Inform the geometry manager that only the given rays are to be used,
        so that internal data size is kept small.
        
        Arguments: 
        idxs - an index array stating which rays of the working bundle
            are active.
        """
        self._idxs = idxs # For slicing ray bundles etc.
        self._backside = N.nonzero(self._backside[idxs])[0]
        
        v = self._working_bundle.get_vertices()[:,idxs]
        d = self._working_bundle.get_directions()[:,idxs]
        p = self._params[idxs]
        del self._params
        
        # Global coordinates on the surface:
        self._global = v + p[None,:]*d
    
    def get_normals(self):
        """
        Report the normal to the surface at the hit point of selected rays in
        the working bundle.
        """
        norms = N.tile(self._working_frame[:3,2].copy()[:,None], (1, len(self._idxs)))
        norms[:,self._backside] *= -1
        return norms
    
    def get_intersection_points_global(self):
        """
        Get the ray/surface intersection points in the global coordinates.
        
        Returns:
        A 3-by-n array for 3 spatial coordinates and n rays selected.
        """
        return self._global
    
    def done(self):
        """
        Discard internal data structures. This should be called after all
        information on the latest bundle's results have been extracted already.
        """
        GeometryManager.done(self)
        if hasattr(self, '_global'):
            del self._global
        if hasattr(self, '_idxs'):
            del self._idxs

class FiniteFlatGM(FlatGeometryManager):
    """
    Calculates intersection points before select_rays(), so that those outside
    the aperture can be dropped, and on select_rays trims it.
    """
    def __init__(self):
        FlatGeometryManager.__init__(self)
    
    def find_intersections(self, frame, ray_bundle):
        """
        Register the working frame and ray bundle, calculate intersections
        and save the parametric locations of intersection on the surface.
        Algorithm taken from [1].
        
        In this class, global- and local-coordinates of intersection points
        are calculated and kept. _global is handled in select_rays(), but
        _local must be taken care off by subclasses.
        
        Arguments:
        frame - the current frame, represented as a homogenous transformation
            matrix stored in a 4x4 array.
        ray_bundle - a RayBundle object with the incoming rays' data.
        
        Returns:
        A 1D array with the parametric position of intersection along each of
            the rays. Rays that missed the surface return +infinity.
        """
        ray_prms = FlatGeometryManager.find_intersections(self, frame, ray_bundle)
        v = self._working_bundle.get_vertices() 
        d = self._working_bundle.get_directions()
        p = self._params
        del self._params
        
        # Global coordinates on the surface:
        self._global = v + p[None,:]*d
        # Local should be deleted by children in their find_intersections.
        self._local = N.dot(N.linalg.inv(self._working_frame),
            N.vstack((self._global, N.ones(self._global.shape[1]))))
        
        return ray_prms
        
    def select_rays(self, idxs):
        """
        Inform the geometry manager that only the given rays are to be used,
        so that internal data size is kept small.
        
        Arguments: 
        idxs - an index array stating which rays of the working bundle
            are active.
        """
        self._idxs = idxs
        self._backside = N.nonzero(self._backside[idxs])[0]
        self._global = self._global[:,idxs].copy()
    
class RectPlateGM(FiniteFlatGM):
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
        
        self._half_dims = N.c_[[width, height]]/2.
        FiniteFlatGM.__init__(self)
        
    def find_intersections(self, frame, ray_bundle):
        """
        Extends the parent flat geometry manager by discarding in advance
        impact points outside a centered rectangle.
        """
        ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
        ray_prms[N.any(abs(self._local[:2]) > self._half_dims, axis=0)] = N.inf
        del self._local
        return ray_prms

class RoundPlateGM(FiniteFlatGM):
    def __init__(self, R):
        """
        Arguments:
        R - the plate's radius
        """
        if R <= 0:
            raise ValueError("Radius must be positive")
        
        self._R = R
        FiniteFlatGM.__init__(self)
    
    def find_intersections(self, frame, ray_bundle):
        """
        Extends the parent flat geometry manager by discarding in advance
        impact points outside a centered circle.
        """
        ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
        ray_prms[N.sum(self._local[:2]**2, axis=0) > self._R**2] = N.inf
        del self._local
        return ray_prms

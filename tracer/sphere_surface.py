# Implements spherical surface 
#
# References:
# [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter1.htm

import numpy as N
from quadric import QuadricGM

class SphericalGM(QuadricGM):
    """
    Implements the geometry of a spherical surface below the xy plane (so 
    that rays going down the Z axis hit). To be used as a base class for
    spherical surfaces that select different hit-points. Otherwise, this is a
    closed sphere.
    """
    def __init__(self, radius=1.):
        """
        Arguments:  
        radius - Set as the sphere's radius
        Private attributes:
        _rad - radius of the sphere, a float 
        """
        QuadricGM.__init__(self)
        self.set_radius(radius)  

    def get_radius(self):
        return self._rad
    
    def set_radius(self, rad):
        if rad <= 0:
            raise ValueError("Radius must be positive")
        self._rad = rad
     
    def _normals(self, hits, directs):
        """
        Finds the normal to the sphere in a bunch of intersection points, by
        taking the derivative and rotating it. Used internally by quadric.
        
        Arguments:                                                                      
        hits - the coordinates of intersections, as an n by 3 array.
        directs - corresponding directions of the hitting rays, n by 3 array.
        """
        c = self._working_frame[:3,3]
        sides = N.sum((c - hits) * directs, axis=1)
        
        normal = (hits - c).T
        normal[:,sides < 0] *= -1
        normal = normal/N.sqrt(N.sum(normal**2, axis=0))
        return normal

    # Ray handling protocol:
    def get_ABC(self, ray_bundle):
        """  
        Determines the variables forming the relevant quadric equation. Used by the quadrics
        class, [1]
        """ 
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self._working_frame[:3,3]
        
        # Solve the equations to find the intersection point:
        A = (d**2).sum(axis=0)
        B = 2*(d*(v - c[:,None])).sum(axis=0)
        C = ((v - c[:,None])**2).sum(axis=0) - self.get_radius()**2
        
        return A, B, C
    
    def mesh(self, resolution):
        """
        Represent the surface as a mesh in local coordinates. Uses spherical-
        coordinates bins, i.e. the points are equally distributed by two angles,
        not by x,y.
        
        Arguments:
        resolution - in points per unit length (so the number of points 
            returned is O(A*resolution**3) for area A)
        
        Returns:
        x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
            coordinate (respectively) of point (i,j) in the mesh.
        """
        rng = N.pi + 1./(self._rad*resolution)
        theta, phi = N.mgrid[
            0:rng:1./(self._rad*resolution),
            0:2*rng:1./(self._rad*resolution)]
        
        x = self._rad * N.sin(theta) * N.cos(phi)
        y = self._rad * N.sin(theta) * N.sin(phi)
        z = self._rad * N.cos(theta)
        
        return x, y, z

class HemisphereGM(SphericalGM):
    def _select_coords(self, coords, prm):
        """
        Select from dual intersections by vetting out rays in the upper
        hemisphere, or if both are below use the default behaviour of choosing
        the first hit.
        """
        select = QuadricGM._select_coords(self, coords, prm) # defaults
        
        coords = N.concatenate((coords, N.ones((2,1,coords.shape[2]))), axis=1)
        local = N.sum(N.linalg.inv(self._working_frame)[None,:,:,None] * \
            coords[:,None,:,:], axis=2)
        bottom_hem = (local[:,2,:] <= 0) & (prm > 0)
        
        select[~N.logical_or(*bottom_hem)] = N.nan
        one_hit = N.logical_xor(*bottom_hem)
        select[one_hit] = N.nonzero(bottom_hem[:,one_hit])[0]
        
        return select
    
    def mesh(self, resolution):
        """
        Represent the surface as a mesh in local coordinates. Uses spherical-
        coordinates bins, i.e. the points are equally distributed by two angles,
        not by x,y.
        
        Arguments:
        resolution - in points per unit length (so the number of points 
            returned is O(A*resolution**3) for area A)
        
        Returns:
        x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
            coordinate (respectively) of point (i,j) in the mesh.
        """
        rng = N.pi + 1./(self._rad*resolution)
        theta, phi = N.mgrid[
            N.pi/2:rng:1./(self._rad*resolution),
            0:2*rng:1./(self._rad*resolution)]

        x = self._rad * N.sin(theta) * N.cos(phi)
        y = self._rad * N.sin(theta) * N.sin(phi)
        z = self._rad * N.cos(theta)
        
        return x, y, z

class CutSphereGM(SphericalGM):
    def __init__(self, radius=1., bounding_volume=None):
        SphericalGM.__init__(self, radius)
        self._bound = bounding_volume
    
    def find_intersections(self, frame, ray_bundle):
        if self._bound is not None:
            self._bound.transform_frame(frame)
        return SphericalGM.find_intersections(self, frame, ray_bundle)
    
    def _select_coords(self, coords, prm):
        """
        Select from dual intersections by checking which of the  impact points
        is contained in a volume defined at object creation time.
        """
        select = SphericalGM._select_coords(self, coords, prm)
        if self._bound is None:
            return select
        
        #in_bd = self._bound.in_bounds(coords) & (prm > 0)
        in_bd = N.array([self._bound.in_bounds(coords[...,c]) for c in xrange(prm.shape[1])]).T
        in_bd &= prm > 0
        select[~N.logical_or(*in_bd)] = N.nan
        one_hit = N.logical_xor(*in_bd)
        select[one_hit] = N.nonzero(in_bd[:,one_hit])[0]
        
        return select


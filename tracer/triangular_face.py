# A geometry manager for a flat surface with a triangular boundary. It's not
# much different than the GMs in the flat_surface module, but it's important
# enough for the case of a triangulated surface, that it gets its own module.
# 
# References:
# [1] http://en.wikipedia.org/wiki/Barycentric_coordinates_(mathematics)
# [2] http://softsurfer.com/Archive/algorithm_0105/algorithm_0105.htm

import numpy as np
from .flat_surface import FiniteFlatGM

class TriangularFace(FiniteFlatGM):
    """
    Trim the infinite flat surface with a triangle. The triangle is formed by
    the origin, and two user-defined points on the XY plane. the plane normal
    is selected to for a right-handed systen with the two points in the order
    they were given.
    """
    
    def __init__(self, verts):
        """
        Arguments:
        verts - a 3x2 array, each column a triangle vertex, in CCW order from
            the origin, on the XY plane. The 3rd vertex is the origin.
        """
        FiniteFlatGM.__init__(self)
        self.set_vertices(verts)
    
    def set_vertices(self, verts):
        """
        replace the defined vertices with new ones.
        """
        self._verts = verts
    
    def find_intersections(self, frame, ray_bundle):
        """
        Register the working frame and ray bundle, calculate intersections
        and save the parametric locations of intersection on the surface.
        plane intersection is done in the base class, triangle trimming done
        here.
        
        In this class, global coordinates of intersection points
        are calculated and kept. _global is handled in select_rays().
        
        Arguments:
        frame - the current frame, represented as a homogenous transformation
            matrix stored in a 4x4 array.
        ray_bundle - a RayBundle object with the incoming rays' data.
        
        Returns:
        A 1D array with the parametric position of intersection along each of
            the rays. Rays that missed the surface return +infinity.
        """
        ray_prms = FiniteFlatGM.find_intersections(self, frame, ray_bundle)
        
        # Transform the charachteristic vertices to the global systenm, then
        # project the global intersection points to get barycentric
        # coordinates, see [1, 2]
        glob_verts = np.dot(frame, np.vstack(( self._verts, np.array([1,1]) )) )
        rel_glob = glob_verts[:3].T - frame[:3,3]
        w = self._global.T - frame[:3,3]
        
        uv = np.dot(self._verts[:,0], self._verts[:,1])
        rel_dots = np.dot(w, rel_glob.T)
        norms_sq = np.sum(self._verts**2, axis=0)
        
        bc = (uv*rel_dots[:,::-1] - norms_sq[::-1]*rel_dots) / \
            (uv**2 - norms_sq[0]*norms_sq[1])
        
        # Use barycentric coordinates for exclusion test.
        outside = np.any(bc < 0, axis=1) | (bc.sum(axis=1) > 1)
        ray_prms[outside] = np.inf
        
        return ray_prms
        
    def mesh(self, resolution=2):
        """
        Represent the surface as a mesh in local coordinates. For the
        triangular face this means twice the head vertex, and each of the other
        two vertices are returned in mesh form.
        
        Arguments:
        resolution - in points per edge (so the number of points 
            returned is O(resolution**2) for area A)
        
        Returns:
        x, y, z - each a 2D array holding in its (i,j) cell the x, y, and z
            coordinate (respectively) of point (i,j) in the mesh.
        """
        if resolution < 2:
            raise ValueError('Resolution must be >= 2')
        
        alpha, beta = np.meshgrid(
            np.linspace(0, 1, resolution), # parameter along two edges
            np.linspace(0, 1, resolution)) # parameter between points on edges
        
        x, y, z = alpha*self._verts[:,1,None,None]*(1 - beta) + \
            alpha*self._verts[:,0,None,None]*beta
        
        return x, y, z

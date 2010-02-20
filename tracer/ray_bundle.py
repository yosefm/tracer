# -*- coding: utf-8 -*-
# References:
# [1] Monte Carlo Ray Tracing, Siggraph 2003 Course 44

import numpy as N

class RayBundle:
    """Contains information about a ray bundle, using equal-length arrays, the
    length of which correspond to the number of rays in a bundle. The number is 
    determined by setting the vertices, so this should be done first when 'filling
    out' a bundle.
    
    Private attributes:
    let n be the number of rays, then for each attribute, .shape[-1] == n.
    _vertices: a 2D array whose each column is the (x,y,z) coordinate of a ray 
        vertex.
    _direct: a 2D array whose each column is the unit vector composed of the 
        direction cosines of each ray.
    _energy: 1D array with the energy carried by each ray.
    _parent: 1D array with the index of the parent ray within the previous ray bundle
    _ref_index: a 1D array with the refraction index of the material a ray is traveling through
    _temp_ref_index: like _ref_index, but used to temporarily store the values that will be
    used in the next iteration of the simulation
    """
    _parent = None
    
    def set_vertices(self,  vert):
        """Sets the starting point of each ray."""
       # if vert.shape != (3,  self.get_num_rays()):
       #     raise ValueError("Number of vertices != number of rays")
        self._vertices = vert
    
    def get_vertices(self):
        return self._vertices
    
    def set_directions(self,  directions):
        """Sets the number of rays as well at the directions"""
        self._direct = directions
    
    def get_directions(self):
        return self._direct
    
    def set_energy(self,  energy):
        self._energy = energy
    
    def get_energy(self):
        return self._energy
    
    def get_num_rays(self):
        return self._direct.shape[1]

    def set_parent(self, index):
        self._parent = index

    def get_parent(self):
        return self._parent

    def set_ref_index(self, ref_index):
        self._ref_index = ref_index
        
    def get_ref_index(self):
        return self._ref_index

    def __add__(self,  added):
        """Merge two energy bundles. return a new bundle with the rays from the 
        two bundles appearing in the order of addition.
        """
        new_parent = N.append(self.get_parent(), added.get_parent())
        newbund = RayBundle()
        newbund.set_directions(N.hstack((self.get_directions(),  added.get_directions())))
        newbund.set_vertices(N.hstack((self.get_vertices(),  added.get_vertices())))
        newbund.set_energy(N.hstack((self.get_energy(),  added.get_energy())))
        newbund.set_parent(new_parent)
        newbund.set_ref_index(N.hstack((self._ref_index, added.get_ref_index()))) 
        return newbund

    @staticmethod
    def empty_bund():
        """Create an empty ray bundle"""
        empty = RayBundle()
        empty_array = N.array([[],[],[]])
        empty.set_directions(empty_array)
        empty.set_vertices(empty_array)
        empty.set_energy(N.array([]))
        empty.set_parent(N.array([]))
        empty.set_ref_index(N.array([]))
        return empty

    def delete_rays(self, selector):
        """Deletes rays"""
        outg = RayBundle()
        outg.set_directions(N.delete(self.get_directions(), selector, axis=1))
        outg.set_vertices(N.delete(self.get_vertices(), selector, axis=1))
        outg.set_energy(N.delete(self.get_energy(), selector))
        outg.set_ref_index(N.delete(self.get_ref_index(), selector))
        if self._parent is not None:
            outg.set_parent(N.delete(self.get_parent(), selector))
         
        return outg 

# Module stuff:
from numpy import random,  linalg as LA
from numpy import c_

from spatial_geometry import general_axis_rotation

def solar_disk_bundle(num_rays,  center,  direction,  radius,  ang_range):
    """Generates a ray bundle emanating from a disk, with each surface element of 
    the disk having the same ray density. The rays all point at directions uniformly 
    distributed between a given angle range from a given direction.
    Setting of the bundle's energy is left to the caller.
    Arguments: num_rays - number of rays to generate.
        center - a column 3-array with the 3D coordinate of the disk's center
        direction - a 1D 3-array with the unit average direction vector for the
            bundle.
        radius - of the disk.
        ang_range - in radians, the maximum deviation from <direction>.
    Returns: a RayBundle object with the above charachteristics set.
    """
    # Diffuse divergence from <direction>:
    # development bassed on eq. 2.12  from [1]
    xi1 = random.uniform(high=2*N.pi, size=num_rays)
    xi2 = random.uniform(size=num_rays)
    theta = N.arcsin(N.sin(ang_range)*N.sqrt(xi2))
    sin_th = N.sin(theta)
    a = N.array((N.cos(xi1)*sin_th, N.sin(xi1)*sin_th , N.cos(theta)))
    # Rotate to a frame in which <direction> is Z:
    perp = N.array([direction[1],  -direction[0],  0])
    if N.all(perp == 0):
        perp = N.array([1.,  0.,  0.])
    perp = perp/LA.norm(perp)
    
    perp_rot = N.array((perp, N.cross(direction, perp), direction)).T
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)
    
    # Locations:
    # See [1]
    xi1 = random.uniform(size=num_rays)
    xi2 = random.uniform(size=num_rays)
    rs = radius*N.sqrt(xi1)
    thetas = 2*N.pi*xi2
    xs = rs * N.cos(thetas)
    ys = rs * N.sin(thetas)
    
    # Rotate locations to the plane defined by <direction>:
    rot = N.vstack((perp,  N.cross(direction,  perp),  direction))
    vertices_local = N.vstack((xs,  ys,  N.zeros(num_rays)))
    vertices_global = N.dot(rot.T,  vertices_local)
    
    rayb = RayBundle()
    rayb.set_vertices(vertices_global + center)
    rayb.set_directions(directions)
    return rayb

def square_bundle(num_rays, center, direction, width):
    perp = N.array([direction[1],  -direction[0],  0])
    if N.all(perp == 0):
        perp = N.array([1.,  0.,  0.])
    perp = perp/N.linalg.norm(perp)
    rot = N.vstack((perp,  N.cross(direction,  perp),  direction))
    directions = N.tile(direction[:,None], (1, num_rays))
    range = N.s_[-width:width:float(2*width)/N.sqrt(num_rays)]
    xs, ys = N.mgrid[range, range]
    vertices_local = N.array([xs.flatten(),  ys.flatten(),  N.zeros(len(xs.flatten()))])
    vertices_global = N.dot(rot,  vertices_local)
    
    rayb = RayBundle()
    rayb.set_vertices(vertices_global + center)
    rayb.set_directions(directions)
    return rayb


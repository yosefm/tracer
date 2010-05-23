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
    def __init__(self, vertices=None, directions=None, energy=None,
        parents=None, ref_index=None):
        """
        Initialize the ray bundle as empty or using the given arrays.
        """
        if vertices is not None:
            self.set_vertices(vertices)
        if directions is not None:
            self.set_directions(directions)
        if energy is not None:
            self.set_energy(energy)
        if parents is not None:
            self.set_parent(parents)
        if ref_index is not None:
            self.set_ref_index(ref_index)
    
    def set_vertices(self,  vert):
        """Sets the starting point of each ray."""
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
    
    def inherit(self, selector=N.s_[:], vertices=None, direction=None, energy=None,
        parents=None, ref_index=None):
        """
        Create a bundle with some ray properties given, and  unspecified
        properties copied from this bundle, at the places noted by `selector`.
        
        Arguments:
        selector - array of ray indices in the current bundle to use, must be
            the same length as the number of rays in the given properties
        vertices, direction, energy, parents, ref_index - set the corresponding
            properties instead of using this bundle.
        """
        if vertices is None and hasattr(self, '_vertices'):
            vertices = self.get_vertices()[:,selector]
        if direction is None and hasattr(self, '_direct'):
            direction = self.get_directions()[:,selector]
        if energy is None and hasattr(self, '_energy'):
            energy = self.get_energy()[selector]
        if parents is None and hasattr(self, '_parent'):
            parents = self.get_parent()[selector]
        if ref_index is None and hasattr(self, '_ref_index'):
            ref_index = self.get_ref_index()[selector]
        
        return RayBundle(vertices, direction, energy, parents, ref_index)

    def __add__(self,  added):
        """Merge two energy bundles. return a new bundle with the rays from the 
        two bundles appearing in the order of addition.         
        
        Arguments:
        added - a RayBundle instance to concatenate with this one.
        """
        newbund = RayBundle()
        
        if hasattr(self, '_direct') and hasattr(added, '_direct'):
            newbund.set_directions(N.hstack((self.get_directions(),  added.get_directions())))
        if hasattr(self, '_vertices') and hasattr(added, '_vertices'):
            newbund.set_vertices(N.hstack((self.get_vertices(),  added.get_vertices())))
        if hasattr(self, '_energy') and hasattr(added, '_energy'):
            newbund.set_energy(N.hstack((self.get_energy(),  added.get_energy())))
        if hasattr(self, '_parent') and hasattr(added, '_parent'):
            new_parent = N.append(self.get_parent(), added.get_parent())
            newbund.set_parent(new_parent)
        if hasattr(self, '_ref_index') and hasattr(added, '_ref_index'):
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
        inherit_select = N.delete(N.arange(self.get_num_rays()), selector)
        outg = self.inherit(inherit_select)
        if hasattr(self, '_parent'):
            outg.set_parent(N.delete(self.get_parent(), selector))
         
        return outg 


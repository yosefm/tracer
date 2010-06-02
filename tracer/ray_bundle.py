# -*- coding: utf-8 -*-

import numpy as N

class RayBundle:
    """
    Contains information about a ray bundle, using equal-length arrays, the
    length of which correspond to the number of rays in a bundle. Each array
    represents one trait of the ray. Of those, at least the vertices and
    directions should be set; for most of the optics managers (see 
    tracer.optics_callables) that come with Tracer, the energy should also be
    set, and for the refractive ones, the current refractive index of the
    volume in which the ray is traveling.
    
    The ray bundle also has a concept of its parent rays - an array where for
    each ray, an index into another bundle of rays is stored. This is used by
    optics managers and the tracer engine to keep track of progression of the
    source rays through reflections and refractions.
    
    For examples on how to create ray bundles, see examples/, tracer.sources, 
    and most of the test-suite.
    """
    
    # Private attributes:
    # let n be the number of rays, then for each attribute, .shape[-1] == n.
    # _vertices - a 2D array whose each column is the (x,y,z) coordinate of a ray 
    #     vertex.
    # _direct - a 2D array whose each column is the unit vector composed of the 
    #     direction cosines of each ray.
    # _energy - 1D array with the energy carried by each ray.
    # _parent - 1D array with the index of the parent ray within the previous ray bundle
    # _ref_index - a 1D array with the refraction index of the material a ray is traveling through
    
    def __init__(self, vertices=None, directions=None, energy=None,
        parents=None, ref_index=None):
        """
        Initialize the ray bundle as empty or using the given arrays. Let n be
        the number of rays, then for each of the arguments, .shape[-1] == n.
        
        Arguments:
        vertices - each column is the (x,y,z) coordinets of a ray's vertex.
        directions - each column is the unit vector composed of the direction
            cosines of each ray.
        energy - each cell has the energy carried by the corresponding ray.
        parents - (not for source rays, for use in optics managers etc.) the 
            index of the parent ray within the ray bundle used to create this
            bundle.
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
        """Sets the starting point of each ray. See __init__()"""
        self._vertices = vert
    
    def get_vertices(self):
        """Returns the starting points of each ray, as a (3,n) array."""
        return self._vertices
    
    def set_directions(self,  directions):
        """Sets the directions of rays in the bundle. See __init__() for the format."""
        self._direct = directions
    
    def get_directions(self):
        """Returns the directions of rays in the bundle."""
        return self._direct
    
    def set_energy(self,  energy):
        """energy - an array with the energy carried by each ray."""
        self._energy = energy
    
    def get_energy(self):
        """Returns an array with the energy carried by each ray in the bundle."""
        return self._energy
    
    def get_num_rays(self):
        """
        Returns the number of rays in the bundle. Assumes that the mandatory
        attributes were set (vertices, directions).
        """
        return self._direct.shape[1]

    def set_parent(self, index):
        """
        Sets the array of indices into a parent bundle. If you're setting
        this and you're not writing an optics manager, you're probably doing
        something wrong.
        """
        self._parent = index

    def get_parent(self):
        """
        Returns the list of parents of each ray in the bundle. It's up to you
        to know which bundle it is referring to.
        """
        return self._parent

    def set_ref_index(self, ref_index):
        """
        Sets the array holding the refractive index of the volume each ray is
        traveling in.
        """
        self._ref_index = ref_index
        
    def get_ref_index(self):
        """
        Returns the array holding the refractive index of the volume each ray is
        traveling in.
        """
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
        """
        Merge two ray bundles. return a new bundle with the rays from the
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
        """
        Create an empty ray bundle - that is, a ray whose attributes are fully
        set, but to empty arrays of correct size.
        """
        empty = RayBundle()
        empty_array = N.array([[],[],[]])
        empty.set_directions(empty_array)
        empty.set_vertices(empty_array)
        empty.set_energy(N.array([]))
        empty.set_parent(N.array([]))
        empty.set_ref_index(N.array([]))
        return empty

    def delete_rays(self, selector):
        """
        Create a new ray bundle which copies this bundle, except in that rays
        denoted by ``selector`` are not copied. Basically equivalent to using
        ``inherit()``, with an inverted selector and no other arguments.
        """
        inherit_select = N.delete(N.arange(self.get_num_rays()), selector)
        outg = self.inherit(inherit_select)
        if hasattr(self, '_parent'):
            outg.set_parent(N.delete(self.get_parent(), selector))
         
        return outg 

def concatenate_rays(bundles):
    """
    Take a list of bundles and merge them into one bundle.
    
    Arguments:
    bundles - a list of RayBundle objects, all with the same set of attributes
        set.
    
    Returns:
    A RayBundle object with all attributes that are set in the first bundle.
    """
    if len(bundles) == 0:
        return RayBundle.empty_bund()
    
    newbund = RayBundle()

    if hasattr(bundles[0], '_direct'):
        newbund.set_directions(N.hstack([b.get_directions() for b in bundles]))
    if hasattr(bundles[0], '_vertices'):
        newbund.set_vertices(N.hstack([b.get_vertices() for b in bundles]))
    if hasattr(bundles[0], '_energy'):
        newbund.set_energy(N.hstack([b.get_energy() for b in bundles]))
    if hasattr(bundles[0], '_parent'):
        newbund.set_parent(N.hstack([b.get_parent() for b in bundles]))
    if hasattr(bundles[0], '_ref_index'):
        newbund.set_ref_index(N.hstack([b.get_ref_index() for b in bundles]))

    return newbund


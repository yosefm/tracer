# -*- coding: utf-8 -*-

import numpy as N
import types

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
        parents=None, ref_index=None, **kwds):
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
        ref_index - each cell has the refractive index of the medium in which
            the corresponding ray travels.
        kwds - more ray properties, each property named by the keyword, with a
            value whose shape[-1] == n.
        """
        base_vals = {
            'vertices': vertices,
            'directions': directions,
            'energy': energy,
            'parents': parents,
            'ref_index': ref_index}
        base_vals.update(kwds)
        
        self._check_attr = [] # Attrs to look for when concatenating bundles
        for n, v in base_vals.iteritems():
            self._create_property(n, v)
    
    def _create_property(self, propname, init_val):
        """
        Add a property of a ray, expected to be an array whose shape[-1] ==
        self.get_num_rays(). The property will be considered for all 
        ray-munging activities (add, inherit, etc.).
        
        Creates the method get_<propname>(self, selector=None). If selector is
        given, it will return only the rays whose index is selected. Also
        creates set_<propname>(self, value, selector=None) which sets either
        the value for all rays or just for the selected rays.
        
        Arguments:
        propname - a string, should be a valid Python identifier.
        init_val - the initial value for the property.
        """
        attr = '_' + propname
        self._check_attr.append(attr)
        
        def getter(self, selector=None):
            if selector is None:
                return self.__dict__[attr]
            else:
                return self.__dict__[attr][...,selector]
        
        def setter(self, new_val, selector=None):
            if selector is None:
                self.__dict__[attr] = new_val
            else:
                self.__dict__[attr][...,selector] = new_val
        
        self.__dict__['get_' + propname] = \
            types.MethodType(getter, self, self.__class__)
        self.__dict__['set_' + propname] = \
            types.MethodType(setter, self, self.__class__)
        
        if init_val is not None:
            self.__dict__['set_' + propname](init_val)
    
    def has_property(self, propname):
        """
        Checks whether the looked-after property ``propname`` exists for this
        bundle.
        """
        return ('_' + propname in self._check_attr)
    
    def get_num_rays(self):
        """
        Returns the number of rays in the bundle. Assumes that the mandatory
        attributes were set (vertices, directions).
        """
        return self._vertices.shape[1]
    
    def inherit(self, selector=N.s_[:], vertices=None, direction=None, energy=None,
        parents=None, ref_index=None, **kwds):
        """
        Create a bundle with some ray properties given, and  unspecified
        properties copied from this bundle, at the places noted by `selector`.
        
        Arguments:
        selector - array of ray indices in the current bundle to use, must be
            the same length as the number of rays in the given properties
        vertices, direction, energy, parents, ref_index - set the corresponding
            properties instead of using this bundle.
        kwds - the same for properties that were created during ray bundle
            construction in addition to the basic set of properties.
        """
        base_vals = dict((attr[1:], None) for attr in self._check_attr)
        base_vals.update({
            'vertices': vertices,
            'directions': direction,
            'energy': energy,
            'parents': parents,
            'ref_index': ref_index})
        base_vals.update(kwds)
        
        for p, v in base_vals.iteritems():
            if base_vals[p] is None and hasattr(self, '_' + p):
                base_vals[p] = self.__dict__['get_' + p](selector)
        
        return RayBundle(**base_vals)

    def __add__(self,  added):
        """
        Merge two ray bundles. return a new bundle with the rays from the
        two bundles appearing in the order of addition.
        
        Arguments:
        added - a RayBundle instance to concatenate with this one.
        """
        newbund = RayBundle()
        
        for attr in self._check_attr:
            if hasattr(self, attr) and hasattr(added, attr):
                # Create the property if its not in the base set:
                if not newbund.has_property(attr):
                    newbund._create_property(attr[1:], None)
                
                # Do addition.
                newbund.__dict__['set' + attr](N.hstack((
                    self.__dict__['get' + attr](),
                    added.__dict__['get' + attr]() )) )
        
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
        empty.set_parents(N.array([]))
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
    
    for attr in bundles[0]._check_attr:
        if hasattr(bundles[0], attr):
            # The new bundle only starts with the basic set, here we equalize:
            if not newbund.has_property(attr):
                newbund._create_property(attr[1:], None)
            
            # This is the actual concatenation:
            getter = 'get' + attr
            newbund.__dict__['set' + attr](N.hstack(
                [b.__dict__[getter]() for b in bundles]))
    
    return newbund


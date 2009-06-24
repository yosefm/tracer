# Implements a tracer engine class

import numpy as N
from flat_surface import FlatSurface 
from ray_bundle import RayBundle
from ray_bundle import solar_disk_bundle
import pylab as P
import pdb

class TracerEngine():
    """
    Tracer Engine implements that actual ray tracing. It keeps track of the number
    of objects, and determines which rays intersected which object.
    """
    def __init__(self,objects):
        """
        Arguments:
        objects - a list of all the objects
        tree - an empty list to be used to track parent rays and child rays. When 
        a ray branches, it simply means that both of the child rays point back to 
        the same index representing that same parent ray
        """
        self.objects = objects
        self.tree = []

    def intersect_ray(self, bundle):
        """
        Returns a boolean array indicating whether there was a hit or a miss, organized
        such that each row of the array matches up to the hits or misses on a single object
        """
        # If there is only a single object, don't need to find minimum distance
        if len(self.objects) == 1:
            stack = [~N.isinf(self.objects[0].register_incoming(bundle))]
            
        else:
            stack = []
            objs_hit = []
            # Bounce rays off each object
            for obj in self.objects:
                stack.append(obj.register_incoming(bundle))
                objs_hit.append(obj)
            stack = N.array(stack)
            objs_hit = N.array(objs_hit) 
            #print stack

            # Raise an error if any of the parameters are negative
            if (stack < 0).any():
                raise ValueError("Parameters must all be positive")

            # If parameter == 0, ray does not actually hit object, but originates from there; 
            # so it should be ignored in considering intersections 
            if (stack == 0).any():
                zeros = N.where(stack == 0)
                stack[zeros] = N.inf

            # Find the smallest parameter for each ray, and use that as the final one,
            # returns the indices
            params_index = stack.argmin(axis=0)
       
            for obj in xrange(len(objs_hit)):
                obj_array = N.where(params_index == obj)
                stack[obj][obj_array] = True 
                stack = (stack == True)
        
        return stack

    def ray_tracer(self, bundle, reps):
        """
        Creates a ray bundle or uses a reflected ray bundle, and intersects it with all
        objects, uses intersect_ray()
        Arguments:
        reps - number of times to repeat the simulation (where each simulation represents
        a ray bundle being intersected with a set of objects one time)
        Returns:
        For the time being, returns an array of vertices of the most recent intersections,
        note that the order of the rays within the arrays may change
        """

        energy = bundle.get_energy()
        bund = bundle
        self.store_branch(bund)
        for i in xrange(reps):
            bund.set_parent(N.array(range(bund.get_num_rays())))
            objs_param = self.intersect_ray(bund)
            outg = bundle.empty_bund()
            parent = bund.get_parent()
            for obj in self.objects:
                inters = objs_param[self.objects.index(obj)]
                new_outg = obj.get_outgoing(inters, energy, parent)
                outg = outg + new_outg
                bund = outg 
            self.store_branch(bund)  # stores parent branch for purposes of ray tracking

        return bund.get_vertices()
                      
    def store_branch(self, bundle):
        """
        Stores a tree of ray bundles
        """
        self.tree.append(bundle)

    def track_parent(self, bundle, index):
        """
        Tracks a particular ray from a bundle, given the bundle 
        it is in and its index within that bundle, and returns the original parent
        """
        i = len(self.tree) - 1
        return self.track_parent_helper(bundle, index, i)

    def track_parent_helper(self, bundle, index, i):
        parent_list = bundle.get_parent()
        parent = parent_list[index]
        bundle = self.tree[i]
        while i > 0:
            i = i-1
            self.track_parent_helper(self, bundle, parent, i)
        return parent

    def track_ray(self, bundle, index):
        """
        Tracks a particular ray from a bundle and returns a list of the coordinates
        of all the intersections. Note that the index of the ray to be tracked is 
        its index in the most recent bundle, not the index of the original bundle.
        """
        self.i = len(self.tree) - 1
        self.coords = []
        return self.track_ray_helper(bundle, index)
       
    def track_ray_helper(self, bundle, index):
        parent_list = bundle.get_parent()
        parent = parent_list[index]
        bundle = self.tree[self.i]
        while self.i >= 0:
            self.i = self.i-1
            self.track_ray_helper(bundle, parent)
            self.coords.append(bundle.get_vertices()[:,parent])
        return self.coords

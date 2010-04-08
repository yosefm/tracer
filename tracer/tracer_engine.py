# Implements a tracer engine class

import numpy as N
from ray_bundle import RayBundle
from ray_bundle import solar_disk_bundle

class TracerEngine():
    """
    Tracer Engine implements that actual ray tracing. It keeps track of the number
    of objects, and determines which rays intersected which object.
    """
    def __init__(self, parent_assembly):
        """
        Arguments:
        parent_assembly - the highest level assembly
        
        Attributes:
        _asm - the Assembly instance containing the model to trace through.
        tree - a list  used for track parent rays and child rays. Each element
            of the list is a ray bundle created after one iteration of the 
            tracer. Each bundle contains an array listing the parent ray in the
            previous bundle (see ray_bundle.py). When a ray branches, the child
            rays point back to the same index representing that same parent ray.
            Otherwise, the index of each ray points to the ray in the previous
            branch.
        """
        self._asm = parent_assembly
        
    def intersect_ray(self, bundle, surfaces, objects, surf_ownership, \
        ray_ownership, surf_relevancy):
        """
        Finds the first surface intersected by each ray.
        
        Arguments:
        bundle - the RayBundle instance holding incoming rays.
        ownership - an array with the owning object instance for each ray in the
            bundle, or -1 for no ownership.
        
        Returns:
        stack - an s by r boolean array for s surfaces and r rays, stating
            for each surface i=1..s if it is intersected by ray j=1..r
        owned_rays - same size as stack, stating whether ray j was tested at all
            by surface i
        """
        ret_shape = (len(surfaces), bundle.get_num_rays())
        stack = N.zeros(ret_shape)
        owned_rays = N.empty(ret_shape, dtype=N.bool)
        
        # Bounce rays off each object
        for surf_num in xrange(len(surfaces)):
            owned_rays[surf_num] = ((ray_ownership == -1) | \
                (ray_ownership == surf_ownership[surf_num])) & surf_relevancy[surf_num]
            if not owned_rays[surf_num].any():
                continue
            in_rays = bundle.delete_rays(N.where(~owned_rays[surf_num])[0])
            stack[surf_num, owned_rays[surf_num]] = \
                surfaces[surf_num].register_incoming(in_rays)
        
        # Raise an error if any of the parameters are negative
        if (stack < -1e-16).any():
            raise ValueError("Parameters must all be positive")
        
        # If parameter == 0, ray does not actually hit object, but originates from there; 
        # so it should be ignored in considering intersections 
        if (stack <= 1e-10).any():
            zeros = N.where(stack <= 1e-6)
            stack[zeros] = N.inf

        # Find the smallest parameter for each ray, and use that as the final one,
        # returns the indices.  If an entire column of the stack is N.inf (the ray misses
        # any surfaces), then take that column to be all False
        stack = ((stack == stack.min(axis=0)) & ~N.isinf(stack))
        
        return stack, owned_rays

    def ray_tracer(self, bundle, reps, min_energy, tree=True):
        """
        Creates a ray bundle or uses a reflected ray bundle, and intersects it
        with all objects, uses intersect_ray(). Based on the intersections,
        generates an outgoing ray in accordance with way the incoming ray
        reflects or refracts off any surfaces.
        
        Arguments:
        bundle - the initial incoming bundle
        reps - stop iteration after this many ray bundles were generated (i.e. 
            after the original rays intersected some surface this many times).
        min_energy - the minimum energy the rays have to have continue tracking
            them; rays with a lower energy are discarded. A float.
        tree - if True, register each bundle in self.tree, otherwise only
            register the last bundle.
        
        Returns: 
        A tuple containing an array of vertices and an array of the the direcitons
        of the last outgoing raybundle (note that the vertices of the new bundle are the 
        intersection points of the previous incoming bundle)
        
        NB: the order of the rays within the arrays may change, but they are tracked
        by the ray tree
        """
        self.tree = []
        bund = bundle
        self.store_branch(bund)
        
        # A list of surfaces and their matching objects:
        surfaces = self._asm.get_surfaces()
        objects = self._asm.get_objects()
        num_surfs = len(surfaces)
        
        surfs_per_obj = [len(obj.get_surfaces()) for obj in objects]
        surfs_until_obj = N.hstack((N.r_[0], N.add.accumulate(surfs_per_obj)))
        surf_ownership = N.repeat(N.arange(len(objects)), surfs_per_obj)
        ray_ownership = -1*N.ones(bund.get_num_rays())
        surfs_relevancy = N.ones((num_surfs, bund.get_num_rays()), dtype=N.bool)
        
        for i in xrange(reps):
            front_surf, owned_rays = self.intersect_ray(bund, surfaces, objects, \
                surf_ownership, ray_ownership, surfs_relevancy)
            outg = bundle.empty_bund()
            out_ray_own = []
            new_surfs_relevancy = []
            
            for surf_idx in xrange(num_surfs):
                inters = front_surf[surf_idx, owned_rays[surf_idx]]
                surfaces[surf_idx].select_rays(N.nonzero(inters)[0])
                if not any(inters): 
                    continue
                new_outg = surfaces[surf_idx].get_outgoing()
                
                # Delete rays with negligible energies
                delete = N.where(new_outg.get_energy() <= min_energy)[0] 
                if N.shape(delete)[0] != 0:
                    new_outg = new_outg.delete_rays(delete)
        
                # add the outgoing bundle from each object into a new bundle
                # that stores all the outgoing bundles from all the objects
                outg = outg + new_outg
                
                # Add new ray-ownership information to the total list:
                obj_idx = surf_ownership[surf_idx]
                surf_rel_idx = surf_idx - surfs_until_obj[obj_idx]
                object_owns_outg = objects[obj_idx].own_rays(new_outg, surf_rel_idx)
                out_ray_own.append(N.where(object_owns_outg, obj_idx, -1))
                
                # Add new surface-relevancy information, saying which surfaces
                # of the full list of surfaces must be checked next. This is
                # somewhat memory-intensize and requires optimization.
                surf_relev = N.ones((num_surfs, new_outg.get_num_rays()), dtype=N.bool)
                surf_relev[surf_ownership == obj_idx] = \
                    objects[obj_idx].surfaces_for_next_iteration(new_outg, surf_rel_idx)
                new_surfs_relevancy.append(surf_relev)
            
            bund = outg
            if bund.get_num_rays() == 0:
                # All rays escaping
                break
            
            ray_ownership = N.hstack(out_ray_own)
            surfs_relevancy = N.hstack(new_surfs_relevancy)
            
            if not tree:
                self.tree = [self.tree[0]] # Delete earlier store.
            self.store_branch(bund)  # stores parent branch for purposes of ray tracking
            
        return bund.get_vertices(), bund.get_directions()
    
    def store_branch(self, bundle):
        """
        Stores a tree of ray bundles in the form of a list of ray bundles. From each bundle
        the list of parent indices can be fetched.
        """
        self.tree.append(bundle)

    def get_parents_from_tree(self):
        """
        Returns a list of arrays of the list of parents for each iteration. Each parent
        array contains indices (matching up to the ray), and these indices point back to the
        previous parent array indicating which parent the current ray originated from.
        """  
        tree = []
        for bundle in self.tree[1:]:
            tree.append(bundle.get_parent())
        return tree

    def track_parent(self, bundle, index):
        """
        Tracks a particular ray from a bundle, given the bundle 
        it is in and its index within that bundle, and returns the original parent
        Arguments:
        bundle - the bundle from where to track the ray from
        index - the index into that bundle of which ray to track (an int)
        Returns: the original parent of that ray, from the very first bundle
        """
        i = len(self.tree) - 1
        return self.track_parent_helper(bundle, index, i)

    def track_parent_helper(self, bundle, index, i):
        """
        A helper function to track_parent().
        Arguments:
        bundle
        index
        i - the length of the tree - 1
        Returns: the original parent of that ray, from the very first bundle
        """
        parent_list = bundle.get_parent() # Gets a list of indexes representing the index of the parent ray within the previous bundle
        parent = parent_list[index]  # Gets the index of the parent of the specific ray of interest
        bundle = self.tree[i]  # Gets the parent bundle
        while i > 0:  # Recurse until the function has walked through the whole tree
            i = i-1   
            self.track_parent_helper(bundle, parent, i)
        return parent

    def track_ray(self, bundle, index):
        """
        Tracks a particular ray from a bundle and returns a list of the coordinates
        of all the intersections. Note that the index of the ray to be tracked is 
        its index in the most recent bundle, not the index of the original bundle.
        Arguments:
        bundle - the RayBundle object from which to track the ray from
        index - the index into that bundle of which ray to track specifically
        Returns: A list of the coordinates of all the intersections that particular ray made
        """
        self.i = len(self.tree) - 1
        self.coords = []
        return self.track_ray_helper(bundle, index)
       
    def track_ray_helper(self, bundle, index):
        """
        A helper function to track_ray
        Arguments:
        bundle
        index
        """
        parent_list = bundle.get_parent()
        parent = parent_list[index]
        bundle = self.tree[self.i]
        while self.i >= 0:
            self.i = self.i-1
            self.track_ray_helper(bundle, parent)
            self.coords.append(bundle.get_vertices()[:,parent])
        return self.coords

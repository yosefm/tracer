# Implements a tracer engine class

import numpy as N
from flat_surface import FlatSurface 
from ray_bundle import RayBundle
from ray_bundle import solar_disk_bundle
from receiver import Receiver
import pylab as P

class TracerEngine():
    """
    Tracer Engine implements that actual ray tracing. It keeps track of the number
    of objects, and determines which rays intersected which object.
    """
    def __init__(self,objects,indices):
        """
        Arguments:
        objects - a list of all the objects and receivers
        indices - a list of indices of which objects in the list objects are receivers
        """
        self.objects = objects
        self.indices = indices

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
        for i in xrange(reps):  
            objs_param = self.intersect_ray(bund)
            outg = bundle.empty_bund()
            for obj in self.objects:
                inters = objs_param[self.objects.index(obj)]
                new_outg = obj.get_outgoing(inters, energy, bund.get_parent())
                new_outg.set_energy(energy[:,inters]) 
                outg = outg + new_outg
                bund = outg 
            for index in self.indices:
                receiver = Receiver(self.objects[index])
                receiver.collect_energy(outg)

        for index in self.indices:
            self.objects(index).energy_plot()

        return bund.get_vertices()
                      





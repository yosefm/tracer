# Implements a tracer engine class

import numpy as N
from flat_surface import FlatSurface 
from ray_bundle import RayBundle
from ray_bundle import solar_disk_bundle
import pylab as P

class TracerEngine():
    """
    Tracer Engine implements that actual ray tracing. It keeps track of the number
    of objects, and determines which rays intersected which object.
    """
    def __init__(self,objects):
        """
        Arguments:
        objects - a list of all the objects
        """
        self.objects = objects

    def intersect_ray(self, bundle):
        """
        Returns objs_param, a list of parameters of intersections for each object, with
        misses replaces by infinity. Filters out any intersections along a ray that
        aren't the intersection with the closest object it hits
        """
        # If there is only a single object, don't need to find minimum distance
        if len(self.objects) == 1:
            objs_param = [self.objects[0].register_incoming(bundle)]

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

            # Find the smallest parameter for each ray, and use that as the final one
            params_index = stack.argmin(axis=0)

            objs_param = []
            for obj in xrange(len(objs_hit)):
                obj_array = N.where(params_index == obj)
                obj_row = stack[obj]
                if N.shape(obj_array[0]) != N.shape(obj_row):
                    obj_row[~obj_array[0]] = N.inf
                objs_param.append(obj_row)

        return objs_param

    def bundle_driver(self, bundle, reps):
        """
        Creates a ray bundle or uses a reflected ray bundle, and intersects it with all
        objects, uses intersect_ray()
        Arguments:
        reps - number of times to repeat the simulation (where each simulation represents
        a ray bundle being intersected with a set of objects one time)
        Returns:
        For the time being, it v, an array of vertices of the most recent intersections
        """

        energy = bundle.get_energy()
        bund = bundle
        for i in xrange(reps):  
            objs_param = self.intersect_ray(bund)

            for obj in self.objects:
                obj_hit = objs_param[self.objects.index(obj)]
                inters = ~N.isinf(obj_hit)
                if self.objects.index(obj) == 0:
                    outg = obj.get_outgoing(inters)
                    outg.set_energy(energy[:,inters])
                else:
                    new_outg = obj.get_outgoing(inters)
                    new_outg.set_energy(energy[:,inters]) 
                    outg = outg + new_outg
                                   
        bund = outg 
        return bund.get_vertices()
                      





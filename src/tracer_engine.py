# Implements a tracer engine class

import numpy as N
from flat_surface import FlatSurface 
import ray_bundle
import optics
import spatial_geometry

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

    #def intersect_ray(self,bundle, source):
        """
        Intersects a ray bundle with all the objects and determines which intersection
        occurred closest, and therefore which intersection to consider for further 
        computations
        Arguments:
        bundle - the ray bundle
        objects - list of objects
        source - starting point of the ray
        Returns:
        List of coordinates for each ray intersection
        """
    
        # checks each individual ray against all the objects
        #hits = []
        #for ray in range(bundle.get_num_rays()):
        #    distances = []
        #    coords = []
        #    for obj in self.objects:
        #        # find the distance from the source to the intersection
        #        hit = obj.register_incoming(bundle)  # wrong, need to do each ray individually
        #        dist = N.linalg.norm(source[:,ray] - hit)

        #    distances.append(dist)
        #    coords.append(hit)
        
        #    # determine which intersection occurred closest
        #    i = N.argmin(distances)
        #    hits.append(coords[i])

        #return hits

    # The above is incorrect, what needs to happen:

    """
    1) bounce a bundle off every object individually
    2) for each object, there is now a list of params for where each ray hit
    3) from each of the params per object, choose the smallest distance
    to be the point of intersection (or rather, index to the same point on 
    all the different list of params and take the params at each of those
    indices and compare)
    4) result should be a list of intersections with different objects, but the
    first intersection of each ray
    """

    def intersect_ray(self, bundle, source):
        hits = []
        coords = []
        for obj in self.objects:
            hits.append(obj.register_incoming(bundle).get_vertices())

        for ray in range(bundle.get_num_rays()):
            dists = []
            for list in hits:
                dist = N.linalg.norm(source[:,ray] - list[ray]) 
                dists.append(dist)
        
            i = N.argmin(dists)
            coords.append(hits[i])

        return coords

    def bundle_driver(self,energy, num_rays, center, direction, radius, ang_range):
        """
        Creates a ray bundle or uses a reflected ray bundle, and intersects it with all
        objects, uses intersect_ray()
        """
        # First set of rays is from the initial source
        bund = ray_bundle.solar_disk_bundle(num_rays, center, direction, radius,
                                            ang_range)
        coords = intersect_ray(bund, bund.get_vertices)
        inters = ~N.isinf(coords)

        # After this first hit, all other rays are reflected
        for i in range(100):  #NEED TO HAVE A LIMIT TO THE LOOP
            coords = intersect_ray(outg, outg.get_vertices()) 
            inters = ~N.isinf(coords)
            outg = obj.get_outgoing(inters)
        
        # Non-intersecting rays
        not_v = bund.get_vertices()[:, ~inters]
        not_d = bund.get_directions()[:, ~inters]
        
        # Returning rays
        v = outg.get_vertices()
        d = outg.get_directions()

        # NEEDS TO KEEP TRACK OF ENERGY TOO

    def recieved(self):
        """
        Shows the energy and distribution of rays on the recieving surface
        """
                      


flat1 = FlatSurface()
direction = N.array([0., 0, -1])
center = N.array([0,0,2]).reshape(-1,1)
sun = ray_bundle.solar_disk_bundle(5000,center, direction, 2, N.pi/1000.) 
engine = TracerEngine([flat1])
engine.intersect_ray(sun, sun.get_vertices())



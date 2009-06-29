# Implements spherical mirrored surface 

from surface import UniformSurface
import optics
from ray_bundle import RayBundle
from boundary_shape import BoundarySphere
import numpy as N

class SphereSurface(UniformSurface):
    """
    Implements the geometry of a spherical mirror surface.  
    """
    def __init__(self, center=None, absorptivity=0., radius=1., boundary=None):
        """
        Arguments:
        location of center, rotation, absorptivity - passed along to the base class.
        boundary - boundary shape defining the surface
        mirrored - either 'in' or 'out', for whether the inner or outer surface is mirrored 
        """
        UniformSurface.__init__(self, center, None,  absorptivity)
        self.set_radius(radius)
        self.center = center
        self.boundary = boundary

    def get_radius(self):
        return self._rad

    def get_center(self):
        return self.center
    
    def set_radius(self, rad):
        if rad <= 0:
            raise ValuError("Radius must be positive")
        self._rad = rad

    # Ray handling protocol:
    def register_incoming(self, ray_bundle):
        """
        Deals wih a ray bundle intersecting with a sphere
        """
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self.get_center()
        params = []
        vertices = []
        norm = []

        # Solve the equations to find the intersection point:
        A = (d**2).sum(axis=0)
        B = (2*d*v - c[:,None]).sum(axis=0)
        C = ((v - c[:,None])**2).sum(axis=0) - self.get_radius()**2
        delta = B**2 - 4*A*C
        
        for ray in xrange(n):
            vertex = v[:,ray]

            if (delta[ray]) < 0:
                params.append(N.inf)
                vertices.append(N.empty([3,1]))
                norm.append(N.empty([3,1]))    
                continue
            
            hits = -B[ray] + N.r_[-1, 1]*N.sqrt(delta[ray])/(2*A[ray])
            coords = vertex + d[:,ray]*hits[:,None]

            is_positive = N.where(hits > 0)[0]
        
            # If both are negative, it is a miss
            if N.shape(is_positive) == (0,):
                params.append(N.inf)
                vertices.append(N.empty([3,1]))
                norm.append(N.empty([3,1]))    
                continue
                
            # If both are positive, us the smaller one
            if len(is_positive) == 2:
                param = N.argmin(hits)
                                        
            # If either one is negative, use the positive one
            else:
                param = is_positive[0]
                
            verts = N.c_[coords[param,:]]
            
            # Define normal based on whether it is hitting an inner or
            # an outer surface of the sphere
            dot = N.vdot(c - coords[param,:], d[:,ray])
            normal = ((coords[param,:] - c) if dot <= 0 else  (c - coords[param,:]))[:,None]
            
            # Check if it is hitting within the boundary
            selector = self.boundary.in_bounds(verts)
            if selector[0]:
                params.append(hits[param])
                vertices.append(verts)
                norm.append(normal)
            else:
                params.append(N.inf)
                vertices.append(N.empty([3,1]))
                norm.append(N.empty([3,1]))    
            
        # Storage for later reference:
        self._vertices = N.hstack(vertices)
        self._current_bundle = ray_bundle
        self._norm = N.hstack(norm)
    
        return params

    def get_outgoing(self, selector, energy, parent):
        dirs = optics.reflections(self._current_bundle.get_directions()[:,selector],
                                  self._norm)
        new_parent = parent[selector]

        outg = RayBundle()
        outg.set_vertices(self._vertices[:,selector])
        outg.set_directions(dirs)
        outg.set_energy(energy[:,selector])
        outg.set_parent(new_parent)

        return outg


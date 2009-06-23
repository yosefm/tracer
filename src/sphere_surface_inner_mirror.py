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

        for ray in xrange(n):
            # Solve the equations to find the intersection point:
            A = d[0,ray]**2 + d[1,ray]**2 + d[2,ray]**2  # simpler way to write and solve this?
            B = 2*(d[0,ray]*(v[0,ray] - c[0])
                   +d[1,ray]*(v[1,ray] - c[1])
                   +d[2,ray]*(v[2,ray] - c[2]))
            C = ((v[0,ray] - c[0])**2 
               +(v[1,ray] - c[1])**2
               +(v[2,ray] - c[2])**2 - self.get_radius()**2)

            vertex = v[:,ray]

            if (B**2 - 4*A*C) < 0:
                params.append(N.inf)
            else:
                t0 = (-B - N.sqrt(B**2 - 4*A*C))/(2*A)
                t1 = (-B + N.sqrt(B**2 - 4*A*C))/(2*A)
                coords = N.c_[v[:,ray] + d[:,ray]*t0, v[:,ray] + d[:,ray]*t1]
                hits = N.r_[[t0,t1]]
                
                # Check if it is hitting the inside surface
                for param in xrange(2):
                    dot = N.vdot(c-coords[:,param], N.c_[coords[:,param]-vertex])
                    if dot <= 0 and hits[param] > 0:
                        verts = N.c_[coords[:,param]]
                        normal = N.c_[c-coords[:,param]]
                   
                        # Check if it is hitting within the boundary
                        selector = self.boundary.in_bounds(verts)
                        if selector[0]:
                            params.append(hits[param])
                            vertices.append(verts)
                            norm.append(normal)
                        else:
                            params.append(N.inf)
                            vertices.append(N.empty([3,1]))
                            
        # Storage for later reference:
        n = len(vertices)
        self._vertices = N.array(vertices).reshape(-1,3).T  
        self._current_bundle = ray_bundle
        self._norm = N.array(norm).reshape(-1,3).T
        
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


# Implements spherical surface where the inside is mirrored

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


        for ray in xrange(n):
            # Solve the equations to find the intersection point:
            A = d[0]**2 + d[1]**2 + d[2]**2  # simpler way to write and solve this?
            B = 2*(d[0]*(v[0] - c[0])
                   +d[1]*(v[1] - c[1])
                   +d[2]*(v[2] - c[2]))
            C = ((v[0] - c[0])**2 
               +(v[1] - c[1])**2
               +(v[2] - c[2])**2 - self.get_radius()**2)

            if (B**2 - 4*A*C) < 0:
                params.append(N.inf)

            else:
                t0 = (-B - N.sqrt(B**2 - 4*A*C))/(2*A)
                t1 = (-B + N.sqrt(B**2 - 4*A*C))/(2*A)
                coords = N.c_[v + d*t0, v + d*t1]
            
                # Check if it is hitting the inside surface
                for param in xrange(2):
                    dot = N.vdot(coords[:,param]-c, N.c_[coords[:,param]]-v)
                    if dot >= 0:
                        verts = N.c_[coords[:,param]]
                        
                # Check if it is hitting within the boundary
                selector = self.boundary.in_bounds(verts)
                if selector[0]:
                    params.append(t1)
                else:
                    params.append(N.inf)
            
        return params


    def get_outgoing(self, incoming):
        vertices = 


boundary = BoundarySphere(N.array([0,1.,0]), 1.)
surface = SphereSurface(center=N.array([0,0,0]), boundary=boundary)
bund = RayBundle()
bund.set_vertices(N.c_[[0,0.,0]])
bund.set_directions(N.c_[[1,0,0]])
print surface.register_incoming(bund)

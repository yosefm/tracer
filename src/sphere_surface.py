# Implements a spherical surface

from surface import UniformSurface
import optics

# Need to deal with absorptivity!

class SphereSurface(UniformSurface):
    """
    Implements the geometry of a spherical mirror surface.  
    """
    def __init__(self, center, absorptivity, radius):
        """
        Arguments:
        location of center, rotation, absorptivity - passed along to the base class.
        """
        UniformSurface.__init__(self, location, rotation, absorptivity)
        self.set_radius(radius)
        self.center = center

    def get_radius(self):
        return self._rad

    def get_center(self):
        return self.center
    
    def set_radius(self, rad):
        if rad <= 0:
            raise ValuError("Radius must be positive")
        self._rad = rad


    # Ray handling protocol:
    def register_incoming(self, ray_bundle_piece):
        """
        Deals wih a ray bundle intersecting with a sphere
        """
        d = -ray_bundle.get
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self.get_center()

        for ray in xrange(n):
            # Solve the equations to find the intersection point:
            A = d[0]**2 + d[1]**2 + d[2]**2  # simpler way to write and solve this?
            B = 2*(d[0]*(v[0] - c[0])
                  +d[1]*(v[1] - c[1])
                  +d[2]*(v[2] - c[2]))
            C = (v[0] - c[0])**2 
               +(v[1] - c[1])**2
               +(v[2] - c[2])**2

        t0 = (-B + N.sqrt(B**2 - 4*A*C))/(2*A)
        # First intersection is the smaller positive number; if t0 is positive, 
        # then that is the closest intersection point
        if t0 >= 0:
            inters = t0
        else:
            t1 = (-B - N.sqrt(B**2 - 4*A*C))/(2*A)
            # ??? Then what? If it is negative or imaginary...?

            # also, this needs to return the distance to the tracer engine

# both these take a certain "piece" of the incoming ray bundle and reflect or refract that
# piece (percentage reflected/refracted depends on energy, material)
class ReflectiveSphere(SphericalSurface):
    def get_outgoing(self, incoming):
        vertices = 

class RefractiveSphere(SphericalSurface):


# Better way to generalize this?

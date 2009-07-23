# Implements spherical surface 

from surface import UniformSurface
import optics
from ray_bundle import RayBundle
from boundary_shape import BoundarySphere
import numpy as N

class SphereSurface(UniformSurface):
    """
    Implements the geometry of a spherical surface.  
    """
    def __init__(self, center=None, absorptivity=0., n=1., 
                 radius=1.):
        """
        Arguments:
        location of center, rotation, absorptivity - passed along to the base class.
        boundary - boundary shape defining the surface
        Private attributes:
        _rad - radius of the sphere
        _center - center of the sphere
        _boundary - boundary shape defining the surface
        _temp_center - holds the value of a temporarily transformed center, for use of 
        calculations by trace engine
        _transform - the transformation of the sphere surface into the frame of the parent
        object. Within it's own local coordinate system the sphere is assume to be centered
        about the origin
        """
        UniformSurface.__init__(self, center, None,  absorptivity)
        self.set_radius(radius)
        self._center = N.append(center, N.c_[[1]])
        self._temp_center = self._center
        self._abs = absorptivity
        self._transform = N.hstack((N.array(([1,0,0],[0,1,0],[0,0,1],[0,0,0])), self._center[:,None]))

    def get_radius(self):
        return self._rad

    def get_center(self):
        return self._center
    
    def set_radius(self, rad):
        if rad <= 0:
            raise ValuError("Radius must be positive")
        self._rad = rad
     
    def get_transform(self):
         return self._transform

    def transform_frame(self, transform):
        self._temp_center = N.dot(transform, self._center)
    
    # Ray handling protocol:
    def register_incoming(self, ray_bundle):
        """
        Deals wih a ray bundle intersecting with a sphere
        Arguments:
        ray_bundle - the incoming bundle 
        Returns a 1D array with the parametric position of intersection along
        each ray.  Rays that miss the surface return +infinity
        """ 
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self._temp_center[:3]

        params = []
        vertices = []
        norm = []
        
        # Solve the equations to find the intersection point:
        A = (d**2).sum(axis=0)
        B = 2*(d*(v - c[:,None])).sum(axis=0)
        C = ((v - c[:,None])**2).sum(axis=0) - self.get_radius()**2
        delta = B**2 - 4*A*C

        for ray in xrange(n):
            vertex = v[:,ray]

            if (delta[ray]) < 0:
                params.append(N.inf)
                vertices.append(N.empty([3,1]))
                norm.append(N.empty([3,1]))    
                continue
            
            hits = (-B[ray] + N.r_[-1, 1]*N.sqrt(delta[ray]))/(2*A[ray])
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

            dot = N.vdot(c.T - coords[param,:], d[:,ray])
            normal = ((coords[param,:] - c) if dot <= 0 else  (c - coords[param,:]))[:,None]
            normal = normal/N.linalg.norm(normal)

            # Check if it is hitting within the boundaries
            for boundary in self.parent_object.get_boundaries():
                selector = boundary.in_bounds(verts)
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
    
    def get_outgoing(self, selector, n1, n2):
        """
        Generates a new ray bundle, which is the reflection of the user selected rays out of
        the incoming ray bundle that was previously registered.
        Arguments:
        selector - a boolean array specifying which rays of the incoming bundle are still relevant
        Returns: a new RayBundle object with the new bundle, with vertices where it intersected with the surface, and directions according to the optic laws
        """
        fresnel = optics.fresnel(self._current_bundle.get_directions()[:,selector], self._norm[:,selector], self._abs, self._current_bundle.get_energy()[selector], n1[selector], n2[selector])
        outg = RayBundle()  
        outg.set_vertices(N.hstack((self._vertices[:,selector], self._vertices[:,selector])))
        outg.set_directions(fresnel[0])
        outg.set_energy(fresnel[1])
        outg.set_parent(N.hstack((self._current_bundle.get_parent()[selector], 
                                 self._current_bundle.get_parent()[selector])))
        outg.set_ref_index(N.hstack((self._current_bundle.get_ref_index()[selector],
                                    self._current_bundle.get_ref_index()[selector])))
                
        # Delete rays with negligible energies 
        delete = N.where(outg.get_energy() <= .05)[0]
        if N.shape(delete)[0] != 0:
            outg = outg.delete_rays(delete)

        return outg


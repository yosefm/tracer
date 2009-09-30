# Implements spherical surface 
#
# References:
# [1] http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm

from surface import UniformSurface
import optics
from ray_bundle import RayBundle
from boundary_shape import BoundarySphere
import numpy as N

class QuadricSurface(UniformSurface):
    """
    Implements the geometry of a quadric surface.
    Private attributes:                                                     
        _loc - location of the surface                                             
        _boundary - boundary shape defining the surface                                      
        _temp_loc - holds the value of a temporarily transformed center, for use of          
        calculations by tracer engine                                                        
        _transform - the transformation of the sphere surface into the frame of the parent   
        object. Within it's own local coordinate system the sphere is assume to be centered  
        about the origin                                                                     
        _inner_n & _outer_n - describe the refractive indices on either side of the surface; 
        note that nothing defines the inside or outside of a surface and it is arbitrarily   
        assigned to the surface that is already facing the air                               
    """
    def __init__(self, location=None, rotation=None, absorptivity=0., mirror=True):
        UniformSurface.__init__(self, location, rotation,  absorptivity, mirror)
        self._inner_n = 1.
        self._outer_n = 1.
    
    # Ray handling protocol:
    def register_incoming(self, ray_bundle):
        """
        Deals wih a ray bundle intersecting with the surface; see [1] 
        Arguments:
        ray_bundle - the incoming bundle 
        Returns a 1D array with the parametric position of intersection along
        each ray.  Rays that miss the surface return +infinity
        """ 
        d = ray_bundle.get_directions()
        v = ray_bundle.get_vertices()
        n = ray_bundle.get_num_rays()
        c = self._temp_frame[:3,3]
        
        params = []
        vertices = []
        norm = []
        
        # Gets the relevant A, B, C from whichever quadric surface, see [1]  
        A, B, C = self.get_ABC(ray_bundle)
        
        delta = B**2 - 4*A*C
    
        for ray in xrange(n):
            vertex = v[:,ray]

            if (delta[ray]) < 0:
                params.append(N.inf)
                vertices.append(N.empty([3,1]))
                norm.append(N.empty([3,1]))    
                continue
            
            if A[ray] <= 1e-10: 
                hit = -C[ray]/B[ray]
                hits = N.hstack((hit,hit))
            
            else: hits = (-B[ray] + N.r_[-1, 1]*N.sqrt(delta[ray]))/(2*A[ray])
            coords = vertex + d[:,ray]*hits[:,None]
            
            # Check if it is hitting within the boundaries
            for boundary in self.parent_object.get_boundaries():
                selector = boundary.in_bounds(coords)
                coords = coords[selector]
                hits = hits[selector]
                
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
            
            dot = N.vdot(c.T - coords[param,:], d[:,ray])
            normal = self.get_normal(dot, coords[param,:], c)
            
            params.append(hits[param])
            vertices.append(verts)
            norm.append(normal)
            
        # Storage for later reference:
        self._vertices = N.hstack(vertices)
        self._current_bundle = ray_bundle
        self._norm = N.hstack(norm)
        
        return params
    
    def get_outgoing(self, selector):
        """
        Generates a new ray bundle, which is the reflection of the user selected rays out of
        the incoming ray bundle that was previously registered.
        Arguments:
        selector - a boolean array specifying which rays of the incoming bundle are still relevant
        Returns: a new RayBundle object with the new bundle, with vertices where it intersected with the surface, and directions according to the optic laws
        """
        outg = RayBundle()
        n1 = self._current_bundle.get_ref_index()[selector]
        n2 = self.get_ref_index(n1)
        
        # Temporary backward-compatibility measure:
        outg.set_temp_ref_index(N.hstack((n2, n1)))
        
        fresnel = optics.fresnel(self._current_bundle.get_directions()[:,selector], \
            self._norm[:,selector], self._absorpt, \
            self._current_bundle.get_energy()[selector], n1, n2, self.mirror)

        outg.set_vertices(N.hstack((self._vertices[:,selector], self._vertices[:,selector])))
        outg.set_directions(fresnel[0])
        outg.set_energy(fresnel[1])
        outg.set_parent(N.hstack((N.arange(self._current_bundle.get_num_rays())[selector], 
                                 N.arange(self._current_bundle.get_num_rays())[selector])))
        outg.set_ref_index(N.hstack((self._current_bundle.get_ref_index()[selector],
                                    self._current_bundle.get_ref_index()[selector])))
        
        return outg

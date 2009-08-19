# Implements a specularly reflecting, grey surface.

from numpy import linalg as LA
import numpy as N

from surface import UniformSurface
from ray_bundle import RayBundle
import optics

class FlatSurface(UniformSurface):
    """Implements the geometry of a flat mirror surface. 
    The local coordinates take Z to be the plane normal.
    """
    def __init__(self,  location=None,  rotation=None,  absorptivity=0., width=1.,  height=1., mirror=None):
        """Arguments: 
        location, rotation, absorptivity - passed along to the base class.
        width - dimension along the surface's local x axis.
        height - dimension along the surface's local y axis.
        Attributes:
        _transform - the transformation of the surface into the frame of the parent object  
        object. Within it's own local coordinate system the sphere is assume to be centered  
        about the origin
        _self._temp_frame & self._temp_rotation - the coordinate system of the surface that 
        has been transform into the flobal coordinates. It is used for caclulations, but eh
        original location and rotation of the surface are simply defined in terms of the 
        coordinate of the object containing the surface
        _inner_n & _outer_n - describe the refractive indices on either side of the surface;  
        note that nothing defines the inside or outside of a surface and it is arbitrarily   
        assigned to the surface that is already facing the air                               
        _mirror - indicates if the surface is fully reflective   
        """
        UniformSurface.__init__(self, location, rotation, absorptivity, mirror)
        self.set_width(width)
        self.set_height(height)
        self._abs = absorptivity 
        self._temp_frame = self._transform
        self._inner_n = 1.
        self._outer_n = 1.
        
    def get_width(self):
        return self._w
    
    def get_height(self):
        return self._h
        
    def set_width(self,  w):
        if w <= 0:
            raise ValueError("Width must be positive")
        self._w = w
    
    def set_height(self,  h):
        if h <= 0:
            raise ValueError("Height must be positive")
        self._h = h
      
    # Ray handling protocol:
    def register_incoming(self,  ray_bundle):
        """This is the first phase of dealing with an energy bundle. The surface
        registers the bundle as a reference (so that future indexing into the same
        bundle is understood).
        Arguments: ray_bundle - a RayBundle object with at-least its vertices and 
            directions specified.
        Returns: a 1D array with the parametric position of intersection along each 
            of the rays. Rays that missed the surface return +infinity.
        """
        xy = self._temp_frame[:3,:2]
        d = -ray_bundle.get_directions()
        v = ray_bundle.get_vertices() - self._temp_frame[:3,3][:,None]
        n = ray_bundle.get_num_rays()
                
        # `params` holds the parametric location of intersections along x axis, 
        # y-axis and ray, in that order.
        params = N.empty((3, n))
        for ray in xrange(n):
            # Solve the linear equation system of the intersection point:
            eqns = N.hstack((xy,  d[:, ray][:, None]))
            if LA.det(eqns) != 0:
                params[:, ray] = N.dot(LA.inv(eqns), v[:, ray])
                continue
            # Singular matrix (parallel rays to the surface):
            params[:, ray].fill(N.inf)
    
        # Mark missing rays with infinity:
        missing = (abs(params[0])  > self._w/2.) | (abs(params[1] ) > self._h/2.)
        params[2, missing] = N.inf

        # Takes into account a negative depth
        # Note that only the 3rd row of params is relevant here!
        negative = params[2] < 0
        params[2, negative] = N.Inf
        
        # Storage for later reference:
        self._current_bundle = ray_bundle
        self._current_params = params[:2]

        return params[2]
    
    def get_outgoing(self,  selector):
        """Generates a new ray bundle, which is the reflections/refractions of the
        user-selected rays out of the incoming ray-bundle that was previously 
        registered.
        Arguments: selector - a boolean array specifying which rays of the incoming
            bundle are still relevant.
        Returns: a RayBundle object with the new bundle, with vertices on the panel
            and directions according to optics laws.
        """
        # Note that n1 is a copy for get_ref_index() since assigning n2 changes the value
        # of the current bundle's refractive index 
        outg = RayBundle()
        n1 = self._current_bundle.get_ref_index().copy()
        n2 = self.get_ref_index(self._current_bundle.get_ref_index(), outg, selector)
        
        # A temp rotation and location are used in which the surface has been transformed
        # into the global coordinates. These are used for calculations
        temp_rotation = self._temp_frame[:3,:3]
        temp_location = self._temp_frame[:3,3]

        
        fresnel = optics.fresnel(self._current_bundle.get_directions()[:,selector], temp_rotation[:,2][:,None], self._abs, self._current_bundle.get_energy()[selector], n1[selector], n2[selector], self.mirror)   

        vertices = N.dot(temp_rotation[:, :2],  self._current_params[:, selector]) + \
            temp_location[:, None]

        outg.set_vertices(N.hstack((vertices, vertices)))
        outg.set_directions(fresnel[0])
        outg.set_energy(fresnel[1]) 
        outg.set_parent(N.hstack((N.arange(self._current_bundle.get_num_rays())[selector],
                                 N.arange(self._current_bundle.get_num_rays())[selector])))
        outg.set_ref_index(N.hstack((self._current_bundle.get_ref_index()[selector],
                                    self._current_bundle.get_ref_index()[selector])))
        
        return outg

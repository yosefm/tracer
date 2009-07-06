# Common functions for optics laws, for use by surfaces etc.

import numpy as N
from ray_bundle import RayBundle 

class Optics():
    """A class that performs optics functions on ray bundles
    Attributes: self._bund - the ray bundle to perform the funtions on
    normals - the normals at the points of intersection
    """
    def __init__(self, bundle, normals):
        self.bund = bundle
        self.normals = normals
        self.ray_dirs_unit = self.normalize(self.bund.get_directions())
         
    def fresnel(self, polar, absorptivity, n):
        """Determines what ratio of the ray bundle is reflected and what is refracted, 
        and the performs the appropriate functions on them.
        Arguments: polar - the type of polarisation
        absorptivity - of the material
        n - refraction index of the material the ray is intersecting with 

        Returns:  a 3-row array whose columns are the ray directions of the
          reflected and refracted rays 
        """
        n1 = bund.get_ref_index()
        n2 = n
        foo = N.cos(self.theta_in) 
        bar = N.sqrt(1 - (n1/n2 * N.sin(self.theta_in))**2)

        Rs = ((n1*foo - n2*bar)/(n1*foo + n2*bar))**2 
        Rp = ((n1*bar - n2*foo)/(n1*bar + n2*foo))**2
        
        if polar == 'p': R = Rp 
        elif polar == 's': R = Rs 
        else: R = (Rs + Rp)/2

        reflected = self.reflections()
        if R + absorptivity > .99:
            return reflected
        else:
            index = int(R*len(reflected))
            reflected = reflected[:,:index]  
            refracted = self.refractions(n)
            
            return N.hstack((reflected, refracted[:,index:]))
                                                             
    def normalize(self, ray_dirs):
        """Normalizes the direction vectors"""
        
        ray_dirs_unit = N.empty_like(ray_dirs)
        theta_in = N.empty_like(ray_dirs[1]) 
        for ray in xrange(ray_dirs.shape[1]):
            ray_dirs_unit[:,ray] = ray_dirs[:,ray]/N.linalg.norm(ray_dirs[:,ray])
            theta_in[ray] = -N.arcsin(N.dot(self.normals[:,ray], ray_dirs_unit[:,ray]))
        self.theta_in = theta_in
        return ray_dirs_unit

    def reflections(self):  
        """Generate directions of rays reflecting according to the reflection law.
        Arguments: ray_dirs - a 3-row array whose columns are direction vectors 
              of incoming rays. It is assumed that their projection on the normal is
              negative.
           normals - a 3-row array containing for each ray in ray_dirs, a corresponding
               unit normal to the surface where that ray hit. If <normals> is a 2D 
               column, it will be broadcast to the correct shape.
        Returns: a 3-row array whose columns are the reflected ray directions.
            """
        ray_dirs = self.bund.get_directions()
        vertical = N.empty_like(ray_dirs)
        # The case of one normal for all rays necessitates replication to make 
        # the loop work.
        if self.normals.shape[1] == 1:
            self.normals = N.tile(self.normals, (1, ray_dirs.shape[1]))
        
        for ray in xrange(ray_dirs.shape[1]):
            vertical[:,ray] = N.inner(ray_dirs[:,ray],  self.normals[:,ray]).T*self.normals[:,ray]
        return ray_dirs - 2*vertical
 
    def refractions(self, n):
        """Generates directions of rays refracted according to Snells's law.
        Arguments: ray_dirs_unit, normals - see reflections()  
            coords - the point of intersection of the incoming ray
            ref_index - a tuple containing n1, n2, n3 the refractive index of the 
               material the ray is exiting, passing through, and entering, respectively
            depth - thickness of the material  
        Returns: a tuple containing a 3-row array whose columns are the points where 
               the rays exits the material and a 3-row array whose columns are the 
               refracted ray directions  
        """ 
        normals_unit = self.normalize(self.normals)
        n1n2 = self.bund.get_ref_index()/n
        cos1 = N.vdot(-normals_unit, self.ray_dirs_unit) 
        self.cos2 = N.sqrt(1 - (n1n2**2)*(1 - cos1**2)) 
        ray_dirs = (n1n2*self.ray_dirs_unit) + (n1n2*cos1 - self.cos2)*normals_unit        
        self.bund.set_ref_index(n)
        return ray_dirs  

bund = RayBundle()
bund.set_vertices(N.c_[[0,1.,0]])
bund.set_directions(N.c_[[-1.,-1,0]])
bund.set_energy(N.r_[[1.]])
bund.set_parent(N.r_[[1.]])
bund.set_ref_index(1)
optics = Optics(bund, N.c_[[0,1.,0]])
ans = optics.fresnel('none', 0, 1.5)
print 
print ans
print N.arctan(ans[0]/ans[1])
print N.arcsin(N.sin(N.pi/4)/1.5)
 


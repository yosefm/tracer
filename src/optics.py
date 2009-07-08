# Common functions for optics laws, for use by surfaces etc.

import numpy as N
from ray_bundle import RayBundle 

        
def fresnel(self, vertices, ray_dirs, absorptivity, n):
    """Determines what ratio of the ray bundle is reflected and what is refracted, 
    and the performs the appropriate functions on them.
    Arguments: polar - the type of polarisation
    absorptivity - of the material
    n - refraction index of the material the ray is intersecting with 
    Returns:  a new ray bundle 
    """
    if N.shape(self.ray_dirs_unit)[1] == 0:
        return self.bund.empty_bund()

    n1 = self.bund.get_ref_index()
    n2 = n
    foo = N.cos(self.theta_in) 
    bar = N.sqrt(N.ones_like(self.theta_in) - (n1/n2 * N.sin(self.theta_in))**2)

    Rs = ((n1*foo - n2*bar)/(n1*foo + n2*bar))**2 
    Rp = ((n1*bar - n2*foo)/(n1*bar + n2*foo))**2
        
    R = (Rs + Rp)/2
    T = 1 - (R+absorptivity)

    # Split the ray bundle in two assuming it is refracted
    empty = self.bund.empty_bund()
    self.bund_new = self.bund + empty
        
    refracted = self.refractions(n, T)
    reflected = self.reflections(R)

    return_bund = refracted + reflected

    # Delete rays with negligble energies
    delete = N.where(return_bund.get_energy() <= .05)[0]
    return_bund.delete_rays(delete)
        
    return return_bund
               
def normalize(self, vector):
    """Normalizes the direction vectors"""
    unit = N.empty_like(vector)
    theta_in = N.empty_like(vector[1]) 
    for ray in xrange(vector.shape[1]):
        unit[:,ray] = vector[:,ray]/N.linalg.norm(vector[:,ray])
        theta_in[ray] = -N.arcsin(N.dot(self.normals[:,ray], unit[:,ray]))
    self.theta_in = theta_in
    return unit

def reflections(self, R):  
    """Generate directions of rays reflecting according to the reflection law.
    Arguments: R - the reflectance
    Returns: a new ray bundle
    """
    ray_dirs = self.bund.get_directions()
    vertical = N.empty_like(ray_dirs)
    # The case of one normal for all rays necessitates replication to make 
    # the loop work.
    if self.normals.shape[1] == 1:
        self.normals = N.tile(self.normals, (1, ray_dirs.shape[1]))

    for ray in xrange(ray_dirs.shape[1]):
        vertical[:,ray] = N.inner(ray_dirs[:,ray],  self.normals[:,ray]).T*self.normals[:,ray]
    self.bund.set_directions(ray_dirs - 2*vertical)
    self.bund.set_parent(self.bund.get_parent()) 

    # Sets the energy of the reflected rays based on Fresnel's equations
    energy = R*self.bund.get_energy()
    self.bund.set_energy(energy)
 
    return self.bund

def refractions(self, n, T):
    """Generates directions of rays refracted according to Snells's law.
    Arguments: T - the transmittance 
    n - the refractive index of the material the ray is passing through
    Returns: a new ray bundle
    """ 
    # Sets the energy of the refracted rays based on Fresnel's equations
    energy = T*self.bund_new.get_energy()
    self.bund_new.set_energy(energy) 

    normals_unit = self.normalize(self.normals)
    n1n2 = self.bund_new.get_ref_index()/n

    cos1 = N.vdot(-normals_unit, self.ray_dirs_unit) 
    self.cos2 = N.sqrt(1 - (n1n2**2)*(1 - cos1**2)) 

    ray_dirs = (n1n2.T*self.ray_dirs_unit) + (n1n2*cos1 - self.cos2).T*normals_unit        
        
    # Set new refractive indices since the rays are travelling through a new material
    self.bund_new.set_ref_index(n)
    self.bund_new.set_parent(self.bund_new.get_parent())
    self.bund_new.set_directions(ray_dirs)

    return self.bund_new

'''
bund = RayBundle()
bund.set_vertices(N.c_[[0,1.,0]])
bund.set_directions(N.c_[[-1.,-1,0]])
bund.set_energy(N.r_[[1.]])
bund.set_parent(N.r_[[1.]])
bund.set_ref_index(N.r_[[1.]])
optics = Optics(bund, N.c_[[0,1.,0]], N.r_[[True]])
ans = optics.fresnel('none', 0, 1.5)
print 
print ans
'''
 


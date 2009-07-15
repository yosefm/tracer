# Common functions for optics laws, for use by surfaces etc.

import numpy as N
from ray_bundle import RayBundle 

        
def fresnel(ray_dirs, normals, absorptivity, energy, n1, n2):
    """Determines what ratio of the ray bundle is reflected and what is refracted, 
    and the performs the appropriate functions on them.
    Arguments: polar - the type of polarisation
    absorptivity - of the material
    n - refraction index of the material the ray is intersecting with 
    Returns:  a new ray bundle 
    """
    
    if N.shape(ray_dirs)[1] == 0:
        return ray_dirs, energy

    theta_in = N.empty_like(ray_dirs[1])
    normals = normals*N.ones_like(ray_dirs)  # for flat surfaces which return only a single value for a normal, make the array the same size as ray_dirs
    for ray in xrange(ray_dirs.shape[1]):
        theta_in[ray] = -N.arcsin(N.dot(normals[:,ray], ray_dirs[:,ray]))

    foo = N.cos(theta_in) 
    bar = N.sqrt(1 - (n1/n2 * N.sin(theta_in))**2)

    Rs = ((n1*foo - n2*bar)/(n1*foo + n2*bar))**2 
    Rp = ((n1*bar - n2*foo)/(n1*bar + n2*foo))**2

    R = (Rs + Rp)/2
    T = 1 - (R+absorptivity)
        
    refracted = refractions(n1, n2, T, ray_dirs, normals)
    reflected = reflections(R, ray_dirs, normals)

    ray_dirs = N.hstack((refracted, reflected))
    energy = N.hstack((energy*T, energy*R))
    return ray_dirs, energy  
               
def reflections(R, ray_dirs, normals):  
    """Generate directions of rays reflecting according to the reflection law.
    Arguments: R - the reflectance
    Returns: a new ray bundle
    """
    vertical = N.empty_like(ray_dirs)
    # The case of one normal for all rays necessitates replication to make 
    # the loop work.
    if normals.shape[1] == 1:
        normals = N.tile(normals, (1, ray_dirs.shape[1]))

    for ray in xrange(ray_dirs.shape[1]):
        vertical[:,ray] = N.inner(ray_dirs[:,ray], normals[:,ray]).T*normals[:,ray]
    ray_dirs = ray_dirs - 2*vertical

    return ray_dirs

def refractions(n1, n2, T, ray_dirs, normals):
    """Generates directions of rays refracted according to Snells's law.
    Arguments: T - the transmittance 
    n - the refractive index of the material the ray is passing through
    Returns: a new ray bundle
    """ 
    for ray in xrange(N.shape(normals)[1]):
        cos1 = N.vdot(-normals[:,ray], ray_dirs[:,ray]) 
    
    cos2 = N.sqrt(1 - (n1/n2**2)*(1 - cos1**2)) 
    
    ray_dirs = ((n1/n2).T*ray_dirs) + (n1/n2*cos1 - cos2).T*normals        
    
    # Set new refractive indices since the rays are travelling through a new material
    
    return ray_dirs


 


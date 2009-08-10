# Common functions for optics laws, for use by surfaces etc.

import numpy as N
from ray_bundle import RayBundle 

        
def fresnel(ray_dirs, normals, absorptivity, energy, n1, n2, mirror):
    """Determines what ratio of the ray bundle is reflected and what is refracted, 
    and the performs the appropriate functions on them.
    Arguments: ray_dirs - the directions of the ray bundle
    absorptivity - of the material
    energy - of the ray bundle
    n1 - refraction index of the material the ray is leaving
    n2 - refraction index of the material the ray is entering
    Returns:  a tuple containing the new ray directions and energy
    """ 
    if N.shape(ray_dirs)[1] == 0:
        return ray_dirs, energy
    
    theta_in = N.empty_like(ray_dirs[1])
    normals = normals*N.ones_like(ray_dirs)  # for flat surfaces which return only a single value for a normal, make the array the same size as ray_dirs
    
    # If the surface is a mirror, simply reflect the ray and don't bother with the 
    # other calculations.  However, since fresnel normally doubles the size of the ray
    # bundle, the reflected ray is doubled in size as is the energy. However the energy for
    # the double is zero so that outgoing_ray() will delete those rays.
    if mirror != False: 
        reflected = reflections(N.ones_like(n1), ray_dirs, normals)
        return N.hstack((reflected, reflected)), N.hstack((energy, N.zeros_like(energy)))

    for ray in xrange(ray_dirs.shape[1]): 
        theta_in[ray] = N.arccos(N.abs(N.dot(normals[:,ray], ray_dirs[:,ray])))
        
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
    ray_dirs, normals - passed from fresnel()
    Returns: new ray directions as the result of reflection
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
    n1, n2, ray_dirs, normals - passed from fresnel
    Returns: new ray directions as the result of refraction
    """
    refr_ray_dirs = N.empty_like(ray_dirs)
    cos1 = N.empty(N.shape(normals)[1])
    for ray in xrange(N.shape(normals)[1]):
        cos1[ray] = (N.vdot(normals[:,ray], -ray_dirs[:,ray]))

    cos2 = N.sqrt(1 - ((n1/n2)**2)*(1 - cos1**2)) 

    pos = N.where(cos1 >= 0)[0]
    neg = N.where(cos1 < 0)[0]
    
    refr_ray_dirs[:,pos] = ((n1[pos]/n2[pos]).T*ray_dirs[:,pos]) + (n1[pos]/n2[pos]*cos1[pos] - cos2[pos]).T*normals[:,pos]        
    refr_ray_dirs[:,neg] = ((n1[neg]/n2[neg]).T*ray_dirs[:,neg]) + (n1[neg]/n2[neg]*cos1[neg] + cos2[neg]).T*normals[:,neg]  
    
    return refr_ray_dirs


 


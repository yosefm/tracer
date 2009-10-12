# Common functions for optics laws, for use by surfaces etc.
#
# References:
# [1] http://en.wikipedia.org/wiki/Fresnel_equations
# [2] http://en.wikipedia.org/wiki/Snell%27s_law

import numpy as N
from ray_bundle import RayBundle 

def fresnel(ray_dirs, normals, n1, n2):
    """Determines what ratio of the ray bundle is reflected and what is refracted, 
    and the performs the appropriate functions on them. Based on fresnel's euqations, [1]
    
    Arguments: 
    ray_dirs - the directions of the ray bundle
    normals - a 3 by n array where for each of the n rays in ray_dirs, the 
        normal to the surface at the ray/surface intersection is given.
    n1 - refraction index of the material the ray is leaving
    n2 - refraction index of the material the ray is entering
    
    Returns:  
    R - the reflectance of a homogenously-polarized light ray with the given
        parameters.
    """ 
    theta_in = N.arccos(N.abs((normals*ray_dirs).sum(axis=0)))
    # Factor out common terms in Fresnel's equations:
    foo = N.cos(theta_in) 
    bar = N.sqrt(1 - (n1/n2 * N.sin(theta_in))**2)
    
    Rs = ((n1*foo - n2*bar)/(n1*foo + n2*bar))**2 
    Rp = ((n1*bar - n2*foo)/(n1*bar + n2*foo))**2

    # For now, assume no polarization and that the light contains an equal mix of 
    # s and p polarized light
    # R = ratio of reflected energy, T = ratio refracted (transmittance)
    R = (Rs + Rp)/2
    return R
               
def reflections(ray_dirs, normals):  
    """
    Generate directions of rays reflecting according to the reflection law.
    
    Arguments:
    ray_dirs - a 3 by n array where each column is the i-th of n ray directions
    normals - for each ray, the corresponding norman on the point where the ray
        intersects a surface, also 3 by n array.
    
    Returns: new ray directions as the result of reflection, 3 by n array.
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
    """Generates directions of rays refracted according to Snells's law (in its vector
    form, [2]
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


 


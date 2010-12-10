# Common functions for optics laws, for use by surfaces etc.
#
# References:
# [1] http://en.wikipedia.org/wiki/Fresnel_equations
# [2] http://en.wikipedia.org/wiki/Snell%27s_law
# [3] Warren J. Smith, Modern Optical Engineering, 4th Ed., 2008; p. 208.

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
    vertical = N.sum(ray_dirs*normals, axis=0)*normals # normal dot ray, really
    return ray_dirs - 2*vertical

def refractions(n1, n2, ray_dirs, normals):
    """Generates directions of rays refracted according to Snells's law (in its vector
    form, [2]
    
    Arguments: 
    n1, n2 - respectively the refractive indices of the medium the unrefracted ray
        travels in and of the medium the ray is entering.
    ray_dirs, normals - each a row of 3-component vectors (as an array) with the
        direction of incoming rays and corresponding normals at the points of
        incidence with the refracting surface.
    
    Returns:
    refracted - a boolean array stating which of the incoming rays has not
        undergone total internal reflection.
    refr_dirs - new ray directions as the result of refraction, for the non-TIR
        rays in the input bundle.
    """
    # Broadcast all necessary arrays to the larger size required:
    n = N.broadcast_arrays(n2/n1, ray_dirs[0])[0]
    normals = N.broadcast_arrays(normals, ray_dirs)[0]
    cos1 = (normals*ray_dirs).sum(axis=0)
    refracted = cos1**2 >= 1 - n**2
    
    # Throw away totally-reflected rays.
    cos1 = cos1[refracted]
    ray_dirs = ray_dirs[:,refracted]
    normals = normals[:,refracted]
    n = n[refracted]
    
    refr_dirs = (ray_dirs - cos1*normals)/n
    cos2 = N.sqrt(1 - 1./n**2*(1 - cos1**2))
    refr_dirs += normals*cos2*N.where(cos1 < 0, -1, 1)
    
    return refracted, refr_dirs

def refr_idx_hartmann(wavelength, a, b, c, d, e):
    """
    Calculate a material's refractive index corresponding to each given
    wavelength, using the Hartmann dispersion equation [3]:
    
    n(L) = a + b/(c - L) + d/(e - L)
    
    where L is the wavelength.
    """
    return a + b/(c - wavelength) 

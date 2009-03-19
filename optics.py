# Common functions for optics laws, for use by surfaces etc.

import numpy as N

def reflections(ray_dirs,  normals):
    """Generate directions of rays reflecting according to the reflection law.
    Arguments: ray_dirs - a 3-row array whose columns are direction vectors 
            of incoming rays. It is assumed that their projection on the normal is
            negative.
        normals - a 3-row array containing for each ray in ray_dirs, a corresponding
            unit normal to the surface where that ray hit. If <normals> is a 2D 
            column, it will be broadcast to the correct shape.
    Returns: a 3-row array whose columns are the reflected ray directions.
    """
    vertical = N.empty_like(ray_dirs)
    # The case of one normal for all rays necessitates replication to make the loop work.
    if normals.shape[1] == 1:
        normals = N.tile(normals, (1, ray_dirs.shape[1]))
        
    for ray in xrange(ray_dirs.shape[1]):
        vertical[:,ray] = N.inner(ray_dirs[:,ray],  normals[:,ray]).T*normals[:,ray]
    return ray_dirs - 2*vertical

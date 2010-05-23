"""
This module contains functions to create some frequently-used light sources.
Each function returns a RayBundle instance that represents the distribution of
rays expected from that source.
"""
from numpy import random, linalg as LA
import numpy as N

from ray_bundle import RayBundle
from spatial_geometry import general_axis_rotation

def solar_disk_bundle(num_rays,  center,  direction,  radius,  ang_range):
    """Generates a ray bundle emanating from a disk, with each surface element of 
    the disk having the same ray density. The rays all point at directions uniformly 
    distributed between a given angle range from a given direction.
    Setting of the bundle's energy is left to the caller.
    Arguments: num_rays - number of rays to generate.
        center - a column 3-array with the 3D coordinate of the disk's center
        direction - a 1D 3-array with the unit average direction vector for the
            bundle.
        radius - of the disk.
        ang_range - in radians, the maximum deviation from <direction>.
    Returns: a RayBundle object with the above charachteristics set.
    """
    # Diffuse divergence from <direction>:
    # development bassed on eq. 2.12  from [1]
    xi1 = random.uniform(high=2*N.pi, size=num_rays)
    xi2 = random.uniform(size=num_rays)
    theta = N.arcsin(N.sin(ang_range)*N.sqrt(xi2))
    sin_th = N.sin(theta)
    a = N.array((N.cos(xi1)*sin_th, N.sin(xi1)*sin_th , N.cos(theta)))
    # Rotate to a frame in which <direction> is Z:
    perp = N.array([direction[1],  -direction[0],  0])
    if N.all(perp == 0):
        perp = N.array([1.,  0.,  0.])
    perp = perp/LA.norm(perp)

    perp_rot = N.array((perp, N.cross(direction, perp), direction)).T
    directions = N.sum(perp_rot[...,None] * a[None,...], axis=1)

    # Locations:
    # See [1]
    xi1 = random.uniform(size=num_rays)
    xi2 = random.uniform(size=num_rays)
    rs = radius*N.sqrt(xi1)
    thetas = 2*N.pi*xi2
    xs = rs * N.cos(thetas)
    ys = rs * N.sin(thetas)

    # Rotate locations to the plane defined by <direction>:
    rot = N.vstack((perp,  N.cross(direction,  perp),  direction))
    vertices_local = N.vstack((xs,  ys,  N.zeros(num_rays)))
    vertices_global = N.dot(rot.T,  vertices_local)

    rayb = RayBundle()
    rayb.set_vertices(vertices_global + center)
    rayb.set_directions(directions)
    return rayb

def square_bundle(num_rays, center, direction, width):
    perp = N.array([direction[1],  -direction[0],  0])
    if N.all(perp == 0):
        perp = N.array([1.,  0.,  0.])
    perp = perp/N.linalg.norm(perp)
    rot = N.vstack((perp,  N.cross(direction,  perp),  direction))
    directions = N.tile(direction[:,None], (1, num_rays))
    range = N.s_[-width:width:float(2*width)/N.sqrt(num_rays)]
    xs, ys = N.mgrid[range, range]
    vertices_local = N.array([xs.flatten(),  ys.flatten(),  N.zeros(len(xs.flatten()))])
    vertices_global = N.dot(rot,  vertices_local)

    rayb = RayBundle()
    rayb.set_vertices(vertices_global + center)
    rayb.set_directions(directions)
    return rayb


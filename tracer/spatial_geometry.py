# Various routines for vector geometry, rotations, etc.
# References:
# [1] John J. Craig, Introduction to Robotics, 3rd ed., 2005. 

from math import sin,  cos
import numpy as N

def general_axis_rotation(axis,  angle):
    """Generates a rotation matrix around <axis> by <angle>, using the right-hand
    rule.
    Arguments: 
        axis - a 3-component 1D array representing a unit vector
        angle - rotation counterclockwise in radians around the axis when the axis 
            points to the viewer.
    Returns: A 3x3 array representing the matrix of rotation.
    Reference: [1] p.47
    """
    s = sin(angle); c = cos(angle); v = 1 - c
    add = N.array([[0,          -axis[2], axis[1]],  
                            [axis[2],  0,          -axis[0]], 
                            [-axis[1], axis[0],  0        ] ])
    return N.multiply.outer(axis,  axis)*v + N.eye(3)*c + add*s

def rotation_to_z(vecs):
    """
    Generate a rotation into a frame whose Z zxis points along the direction
    indicated by ``vecs``. The rest of the directions are determined by
    requiring the new X to lie on the XY plane of the original frame, and by
    the right-hand rule.
    
    In the singular case that the required direction is the Z axis, the
    original frame is retained.
    
    Arguments:
    vec - a 3-component 1D array representing a unit direction in 3D space.
    
    Returns:
    A 3x3 array representing the global-to-local rotation to get the desired
        frame.
    """
    vecs = N.atleast_2d(vecs)
    perp = N.hstack((vecs[:,1][:,None],  -vecs[:,0][:,None],
        N.zeros((vecs.shape[0], 1)) ))
    perp[N.all(perp == 0, axis=1)] = N.r_[1.,  0.,  0.]
    perp /= N.sqrt(N.sum(perp**2, axis=1))[:,None]
    perp_rot = N.concatenate((perp[...,None], N.cross(vecs, perp)[...,None],
        vecs[...,None]), axis=2)
    return N.squeeze(perp_rot)

def generate_transform(axis, angle, translation):
    """Generates a transformation matrix                                                      
    Arguments: axis - a 3-component 1D array giving the unit vector to rotate about          
    angle - angle of rotation counter clockwise in radians about the given axis in the 
    parent frame                         
    translation - a 3-component column vector giving the translation along the coordinates
    of the parent object/assembly
    """  
    rot = general_axis_rotation(axis, angle)
    return N.vstack((N.hstack((rot, translation)), N.r_[[0,0,0,1]]))

def rotx(ang):
    """Generate a homogenous trransform for ang radians around the x axis"""
    s = sin(ang); c = cos(ang)
    return N.array([
        [1., 0, 0, 0],
        [0, c,-s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

def roty(ang):
    """Generate a homogenous trransform for ang radians around the y axis"""
    s = sin(ang); c = cos(ang)
    return N.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s,0, c, 0],
        [0, 0, 0, 1]
    ])

def rotz(ang):
    """Generate a homogenous trransform for ang radians around the z axis"""
    s = sin(ang); c = cos(ang)
    return N.array([
        [c,-s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translate(x, y, z):
    """Generate a homogenous transform for translation by x, y, z"""
    return N.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0 ,0, 1, z],
        [0, 0, 0, 1]
    ])

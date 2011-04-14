
import numpy as np
from ..object import AssembledObject
from ..surface import Surface
from ..triangular_face import TriangularFace

class TriangulatedSurface(AssembledObject):
    """
    Represent a set of triangular faces composing a surface.
    """
    
    def __init__(self, vertices, faces, optics, transform=None):
        """
        Create the triangular faces from a list of vertices and the topology
        information. Somewhat like VRML's IndexedFaceSet, only limited to
        triangular faces.
        
        Arguments:
        vertices - an (n,3) array of n 3D points in the object's frame.
        faces - an (n,3) integer array, each row is 3 indices into the
            vertices array, for the 3 vertices of one triangular face.
        optics - the optics manager to assign each surface.
        transform - a 4x4 array representing the homogenous transformation 
            matrix of this object relative to the coordinate system of its 
            container
        """
        face_list = [None]*vertices.shape[0]
        for face_ix, face_vert_idxs in enumerate(faces):
            face_verts = vertices[face_vert_idxs]
            pos = face_verts[0]
            
            # Frame directions: X along first edge, Z along the normal, Y
            # completes a right-handed frame XYZ.
            edges = face_verts[1:] - pos
            x = edges[0]/np.linalg.norm(edges[0])
            z = np.cross(edges[0], edges[1])
            z /= np.linalg.norm(z)
            y = np.cross(z, x)
            rot = np.c_[x, y, z]
            
            edges_local = np.dot(rot.T, edges.T)
            geom = TriangularFace(edges_local)
            face_list[face_ix] = Surface(geom, optics, location=pos, rotation=rot)
        
        AssembledObject.__init__(self, face_list, None, transform)


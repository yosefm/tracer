
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
        self._verts = vertices
        self._topo = faces
        
        pos = vertices[faces[:,0]]
        edges = vertices[faces[:,1:],:] - pos[:,None,:]
        edge_norms = np.sqrt(np.sum(edges**2, axis=2))
        non_degenerate = np.all(abs(edge_norms) > 1e-8, axis=1)
        edges = edges[non_degenerate]
        edge_norms = edge_norms[non_degenerate]
        
        xs = edges[:,0] / edge_norms[:,0,None]
        zs = np.cross(xs, edges[:,1])
        zs /= np.sqrt((zs**2).sum(-1))[:,None]
        non_collinear = np.any(abs(zs) > 1e-6, axis=1)
        
        xs = xs[non_collinear]
        zs = zs[non_collinear]
        ys = np.cross(zs, xs)
        edges = edges[non_collinear]
        
        rots = np.concatenate((xs[...,None], ys[...,None], zs[...,None]), axis=2)
        edges_local = np.sum(rots.transpose(0,2,1)[:,None,...]*\
            edges[:,:,None,:], axis=3)
        
        self._fl = [None]*faces.shape[0]
        real_faces = non_degenerate
        real_faces[~non_colinear] = False
        
        for face_ix in np.nonzero(real_faces)[0]:
            new_face = Surface(TriangularFace(edges_local[face_ix].T), optics,
                location=pos[face_ix], rotation=rots[face_ix])
            self._fl[face_ix] = new_face
        
        face_list = [s for s in self._fl if s is not None]
        AssembledObject.__init__(self, face_list, None, transform)

    def move_vertices(self, vert_ix, new_verts):
        """
        Replaces some vertex positions with new positions, without altering the
        topology. Edits the faces touching the moved vertex. If colinearity or
        degeneracy changes, adds or removes a surface appropriately.
        
        Arguments:
        vert_ix - an array of n indices of the vertices to replace, referes to
            the vertices given to the constructor.
        new_verts - an (n,3) array, row i is a new vertex position for the
            vertex whose index is vert_ix i.
        """
        self._verts[vert_ix] = new_verts
        
        # Update faces:
        update = np.nonzero(np.any(
            np.any(faces[...,None] == vert_ix[None,None,:], axis=-1), axis=-1))[0]
        pos = vertices[faces[update,0]]
        edges = vertices[faces[update,1:],:] - pos[:,None,:]


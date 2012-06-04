
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
        
        face_list = [Surface(TriangularFace(edges_local[face_ix].T), optics,
            location=pos[face_ix], rotation=rots[face_ix]) \
            for face_ix in xrange(xs.shape[0])]
        
        AssembledObject.__init__(self, face_list, None, transform)


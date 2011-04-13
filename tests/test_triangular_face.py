# Unit tests for the tracer.triangular_face module.

import unittest
import numpy as np

from tracer.triangular_face import TriangularFace
from tracer.ray_bundle import RayBundle
from tracer.spatial_geometry import translate, rotx

class TestExclusion(unittest.TestCase):
    def setUp(self):
        x, y = np.mgrid[-1:1:2j, -1:1:3j]
        pos = np.vstack((x.flatten(), y.flatten(), np.ones(6)*10))
        pos[1] -= 10.
        direct = np.tile(np.c_[[0., 10., -10.]], (1, 6))
        self.bund = RayBundle(pos, direct)
    
        verts = np.zeros((3,2))
        verts[1] = 2.
        verts[0] = np.r_[4., -4.]
        
        self.tri = TriangularFace(verts)
    
    def test_default_frame(self):
        """A triangle whose frame is the global frame excludes rays correctly"""
        prm = self.tri.find_intersections(np.eye(4), self.bund)
        
        np.testing.assert_array_equal(np.isfinite(prm),
            np.r_[False, False, True, False, False, True])
    
    def test_transformed(self):
        """A transformed triangle excludes rays correctly"""
        frame = np.dot(translate(0, -1, 0.5), rotx(np.pi/4.))
        prm = self.tri.find_intersections(frame, self.bund)
        
        np.testing.assert_array_equal(np.isfinite(prm),
            np.r_[False, False, True, False, False, True])

class TEstMesh(unittest.TestCase):
    def test_points(self):
        verts = np.zeros((3,2))
        verts[1] = 2.
        verts[0] = np.r_[4., -4.]
        tri = TriangularFace(verts)
        
        x, y, z = tri.mesh(3)
        cx = np.array([[0., -2., -4.], [0., 0., 0.], [0., 2., 4.]])
        cy = np.array([[0., 1., 2.], [0., 1., 2.], [0., 1., 2.]])
        np.testing.assert_array_equal(x, cx)
        np.testing.assert_array_equal(y, cy)
        np.testing.assert_array_equal(z, np.zeros_like(x))

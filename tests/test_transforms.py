import unittest
import numpy as np
from tracer import spatial_geometry as sg

class TestRotateToZ(unittest.TestCase):
    def test_single_vec(self):
        """A rotation into one vector is correct"""
        vec = np.r_[1., 1., 1.]/np.sqrt(3)
        rot = sg.rotation_to_z(vec)
        np.testing.assert_array_almost_equal(rot, np.c_[
            np.r_[1., -1, 0.]/np.sqrt(2), np.r_[1., 1., -2.]/np.sqrt(6), vec])
    
    def test_two_vecs(self):
        """Vectorization of rotation into a vector"""
        vecs = np.vstack((np.r_[1., 1., 1.]/np.sqrt(3), np.r_[1., 0., 0.]))
        rots = sg.rotation_to_z(vecs)
        
        np.testing.assert_array_almost_equal(rots[0], np.c_[
            np.r_[1., -1, 0.]/np.sqrt(2), np.r_[1., 1., -2.]/np.sqrt(6), vecs[0]])
        np.testing.assert_array_almost_equal(rots[1], np.c_[
            [0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


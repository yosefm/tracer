# Test the parabolic dish's additions to the paraboloid.

import unittest
import numpy as N

from tracer.paraboloid import ParabolicDishGM, HexagonalParabolicDishGM
from tracer.ray_bundle import RayBundle
from tracer.surface import Surface
from tracer.spatial_geometry import generate_transform
import tracer.optics_callables as opt

class TestParabolicDish(unittest.TestCase):
    def setUp(self):
        pos = N.zeros((3,4))
        pos[0] = N.r_[0, 0.5, 2, -2]
        pos[2] = 2.
        dir = N.tile(N.c_[[0,0,-1]], (1,4))

        self.bund = RayBundle()
        self.bund.set_vertices(pos)
        self.bund.set_directions(dir)

        self.surf = Surface(ParabolicDishGM(2., 1.), opt.perfect_mirror)
    
    def test_selection_at_origin(self):
        """Simple dish rejects missing rays"""
        misses = N.isinf(self.surf.register_incoming(self.bund))
        N.testing.assert_array_equal(misses, N.r_[False, False, True, True])
    
    def test_transformed(self):
        """Translated and rotated dish rejects missing rays"""
        trans = generate_transform(N.r_[1., 0., 0.], N.pi/4., N.c_[[0., 0., 1.]])
        
        self.surf.transform_frame(trans)
        misses = N.isinf(self.surf.register_incoming(self.bund))
        N.testing.assert_array_equal(misses, N.r_[False, False, True, True])

class TestHexDish(unittest.TestCase):
    def runTest(self):
        pos = N.array([[0, 1.5], [0, -1.5], [1, 0], [-1, 0], [0.1, 0.1], [-0.1, 0.6]])
        bund = RayBundle()
        bund.set_vertices(N.vstack((pos.T, N.ones(pos.shape[0]))))
        bund.set_directions(N.tile(N.c_[[0,0,-1]], (1,6)))
        surf = Surface(HexagonalParabolicDishGM(2., 1.), opt.perfect_mirror)
        
        misses = N.isinf(surf.register_incoming(bund))
        N.testing.assert_array_equal(misses, N.r_[True, True, True, True, False, False])

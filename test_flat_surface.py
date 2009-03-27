import unittest
import math
import numpy as N
from scipy import c_, r_

from flat_surface import FlatSurface
from ray_bundle import RayBundle

class TestFlatSurfaceInterface(unittest.TestCase):
    def runTest(self):
        """Doesn't allow negative width or height"""
        surf = FlatSurface()
        self.assertRaises(ValueError, surf.set_width, -42)
        self.assertRaises(ValueError, surf.set_height, -42)

class TestTraceProtocol(unittest.TestCase):
    def setUp(self):
        self._surf = FlatSurface()
        
        dir = N.array([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]).T / math.sqrt(3)
        position = c_[[0,0,1], [1,-1,1], [1,1,1], [-1,1,1]]
        
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        
    def test_register_incoming(self):
        """A simple bundle is registered correctly"""
        correct_params = r_[[math.sqrt(3)]*3]
        params = self._surf.register_incoming(self._bund)
        
        self.failUnless(params[0] == N.inf)
        N.testing.assert_array_almost_equal(params[1:], correct_params)
        

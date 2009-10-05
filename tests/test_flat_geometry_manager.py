# Test the flat geometry manager for conformance to the required interface

import unittest
import numpy as N
import math

from flat_surface import FlatGeometryManager
from ray_bundle import RayBundle

class TestFlatGeomManagerInterface(unittest.TestCase):
    def setUp(self):
        dir = N.c_[[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]] / math.sqrt(3)
        position = N.c_[[0,0,1], [1,-1,1], [1,1,1], [-1,1,1]]

        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        
    def test_find_intersections(self):
        """The correct parametric locations are found for flat geometry"""
        gm = FlatGeometryManager()
        prm = gm.find_intersections(N.eye(4), self._bund)
        
        self.failUnlessEqual(prm.shape, (4,), 
            "Shape of parametric location array is wrong: " + str(prm.shape))
        N.testing.assert_array_almost_equal(prm, N.sqrt(3))
    
    def test_get_normals(self):
        gm = FlatGeometryManager()
        gm.find_intersections(N.eye(4), self._bund)
        
        n = gm.get_normals(N.ones(4, dtype=N.bool)) # all rays selected
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, 1]], (1,4)))
        
        n = gm.get_normals(N.array([0, 1, 0, 1], dtype=N.bool)) # just two rays selected
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, 1]], (1,2)))

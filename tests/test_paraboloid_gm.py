# Test the paraboloid geometry manager.

import unittest
import numpy as N

from ray_bundle import RayBundle
from paraboloid import Paraboloid

class TestInterface(unittest.TestCase):
    def setUp(self):
        self.num_rays = 10
        dir = N.tile(N.c_[[0, 0, -1]], (1, self.num_rays))
        theta = N.linspace(0, 2*N.pi, self.num_rays, endpoint=False)
        position = N.vstack((N.cos(theta), N.sin(theta), N.ones(self.num_rays)))
        
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)

        self.gm = Paraboloid(a=5., b=5.)
        self.prm = self.gm.find_intersections(N.eye(4), self._bund)
    
    def test_find_intersections(self):
        """The correct parametric locations are found for paraboloid geometry"""
        self.failUnlessEqual(self.prm.shape, (self.num_rays,))
        N.testing.assert_array_almost_equal(self.prm, 0.96)
    
    def test_get_normals(self):
        """Paraboloid surface returns center-pointing normals"""
        n = self.gm.get_normals(N.ones(self.num_rays, dtype=N.bool)) # all rays selected
        N.testing.assert_array_equal(n[-1,0], n[-1,1:])
        N.testing.assert_array_almost_equal(self._bund.get_vertices()[:2],
            -n[:2]/N.sqrt((n[:2]**2).sum(axis=0)))
    
    def test_inters_points_global(self):
        """Paraboloid returns correct intersections"""
        pts = self.gm.get_intersection_points_global(N.ones(self.num_rays, dtype=N.bool))
        N.testing.assert_array_equal(pts[:2], self._bund.get_vertices()[:2])
        N.testing.assert_array_almost_equal(pts[2], 0.04)
        

# Test the paraboloid geometry manager.

import unittest
import numpy as N

from tracer.ray_bundle import RayBundle
from tracer.paraboloid import Paraboloid, ParabolicDishGM, HexagonalParabolicDishGM

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
        self.gm.select_rays(N.arange(self.num_rays))
        n = self.gm.get_normals() # all rays selected
        N.testing.assert_array_equal(n[-1,0], n[-1,1:])
        N.testing.assert_array_almost_equal(self._bund.get_vertices()[:2],
            -n[:2]/N.sqrt((n[:2]**2).sum(axis=0)))
    
    def test_inters_points_global(self):
        """Paraboloid returns correct intersections"""
        self.gm.select_rays(N.arange(self.num_rays))
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_equal(pts[:2], self._bund.get_vertices()[:2])
        N.testing.assert_array_almost_equal(pts[2], 0.04)
        
class TestParabolicDishGM(unittest.TestCase):
    def setUp(self):
        dir = N.c_[[0., 0, -1], [0, 1, -1], [0, 11, -2], [0, 1, 0]]
        dir /= N.sqrt(N.sum(dir**2, axis=0))
        position = N.c_[[0., 0, 1], [0, -1, 1], [0, -11, 2], [0, 1, 1]]

        bund = RayBundle()
        bund.set_vertices(position)
        bund.set_directions(dir)
        self.bund = bund
        
        self.correct = N.r_[1., N.sqrt(2), N.sqrt(11**2 + 4)]
        
    def test_circular_aperture(self):
        """Rays that intersect a paraboloid above the cut plane handled correctly"""
        gm = ParabolicDishGM(diameter=10., focal_length=12.)
        prm = gm.find_intersections(N.eye(4), self.bund)
        
        N.testing.assert_array_almost_equal(prm[:3], self.correct)
        self.failUnless(prm[3] == N.inf)
    
    def test_hex_aperture(self):
        """Rays intersecting a paraboloid with hex aperture handled correctly"""
        gm = HexagonalParabolicDishGM(diameter=10., focal_length=12.)
        prm = gm.find_intersections(N.eye(4), self.bund)
        
        N.testing.assert_array_almost_equal(prm[:3], self.correct)
        self.failUnless(prm[3] == N.inf)


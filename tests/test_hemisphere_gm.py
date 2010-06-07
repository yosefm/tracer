# Test the paraboloid geometry manager.

import unittest
import numpy as N

from tracer.ray_bundle import RayBundle
from tracer.sphere_surface import HemisphereGM

class TestInterface(unittest.TestCase):
    def setUp(self):
        self.num_rays = 10
        dir = N.tile(N.c_[[0, 0, -1]], (1, self.num_rays))
        theta = N.linspace(0, 2*N.pi, self.num_rays, endpoint=False)
        position = N.vstack((N.cos(theta), N.sin(theta), N.ones(self.num_rays)))
        self._bund = RayBundle(position, dir)
        
        self.gm = HemisphereGM(radius=2.)
        self.prm = self.gm.find_intersections(N.eye(4), self._bund)
    
    def test_find_intersections(self):
        """The correct parametric locations are found for hemisphere geometry"""
        self.failUnlessEqual(self.prm.shape, (self.num_rays,))
        N.testing.assert_array_almost_equal(self.prm, 1 + 2*N.sin(N.pi/3))
    
    def test_get_normals(self):
        """Hemisphere surface returns center-pointing normals"""
        self.gm.select_rays(N.arange(self.num_rays))
        n = self.gm.get_normals()
        N.testing.assert_array_almost_equal(n[-1,0], n[-1,1:])
        N.testing.assert_array_almost_equal(self._bund.get_vertices()[:2],
            -n[:2]/N.sqrt((n[:2]**2).sum(axis=0)))
    
    def test_inters_points_global(self):
        """Hemisphere returns correct intersections"""
        self.gm.select_rays(N.arange(self.num_rays))
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_equal(pts[:2], self._bund.get_vertices()[:2])
        N.testing.assert_array_almost_equal(pts[2], -2*N.sin(N.pi/3))
    
    def test_mesh(self):
        """The HemisphereGM mesh represents the lower hemisphere only"""
        x, y, z = self.gm.mesh(10)
        self.failUnless(N.all(z <= 1e-15))
        self.failIf(N.any(x**2 + y**2 > 4.0001))


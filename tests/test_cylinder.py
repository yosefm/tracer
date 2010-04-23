# Test that the infinite cylinder geometry does what it should.

import unittest
import numpy as N

from tracer.ray_bundle import RayBundle
from tracer.cylinder import InfiniteCylinder
from tracer.spatial_geometry import rotx

class TestInfiniteCylinder(unittest.TestCase):
    def setUp(self):
        # Two rays inside, two outside; two horizontal, two slanted.
        pos = N.c_[[0., 0., 0.], [0., 0., 0.], [0., 1., 0.], [0., 1., 0.]]
        dir = N.c_[[0., 1., 0.], [0., 1., 1.], [0.,-1., 0.], [0.,-1., 1.]]
        dir /= N.sqrt(N.sum(dir**2, axis=0))
        
        self.bund = RayBundle()
        self.bund.set_vertices(pos)
        self.bund.set_directions(dir)
        
        self.gm = InfiniteCylinder(diameter=1.)
    
    def test_vertical(self):
        """Cylinder parallel to the Z axis"""
        correct_prm = N.r_[0.5, N.sqrt(0.5), 0.5, N.sqrt(0.5)]
        prm = self.gm.find_intersections(N.eye(4), self.bund)
        N.testing.assert_array_almost_equal(prm, correct_prm)
        
        self.gm.select_rays(N.arange(4))
        
        correct_norm = N.c_[[0.,-1., 0.], [0.,-1., 0.], [0.,1., 0.], [0.,1., 0.]]
        norm = self.gm.get_normals()
        N.testing.assert_array_almost_equal(norm, correct_norm)
        
        correct_pts = N.c_[
            [0., 0.5, 0.], [0., 0.5, 0.5], 
            [0., 0.5, 0.], [0., 0.5, 0.5]]
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_almost_equal(pts, correct_pts)
    
    def test_rotated(self):
        """Rotate cylinder by 30 degrees"""
        frame = rotx(N.pi/6)
        
        ydist = 0.5/N.cos(N.pi/6)
        # Ray parameter of slanted rays using sine law:
        p1 = N.sin(2*N.pi/3) * (1 - ydist) / N.sin(N.pi/12)
        p2 = N.sin(N.pi/3) * ydist / N.sin(5*N.pi/12)
        correct_prm = N.r_[ydist, p2 , 1 - ydist, p1]
        prm = self.gm.find_intersections(frame, self.bund)
        N.testing.assert_array_almost_equal(prm, correct_prm)

        self.gm.select_rays(N.arange(4))
        
        outwrd = N.r_[0., N.cos(N.pi/6), 0.5]
        correct_norm = N.c_[-outwrd, -outwrd, outwrd, outwrd]
        norm = self.gm.get_normals()
        N.testing.assert_array_almost_equal(norm, correct_norm)

        correct_pts = N.c_[
            [0., ydist, 0.], [0., p2*N.sqrt(0.5), p2*N.sqrt(0.5)],
            [0., ydist, 0.], [0., 1 - p1*N.sqrt(0.5), p1*N.sqrt(0.5)]]
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_almost_equal(pts, correct_pts)

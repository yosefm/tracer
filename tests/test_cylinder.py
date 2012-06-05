# Test that the infinite cylinder geometry does what it should.

import unittest
import numpy as N

from tracer.ray_bundle import RayBundle
from tracer.cylinder import InfiniteCylinder, FiniteCylinder
from tracer.spatial_geometry import rotx

class TestInfiniteCylinder(unittest.TestCase):
    def setUp(self):
        # Two rays inside, two outside; two horizontal, two slanted.
        pos = N.c_[[0., 0., 0.], [0., 0., 0.], [0., 1., 0.], [0., 1., 0.]]
        dir = N.c_[[0., 1., 0.], [0., 1., 1.], [0.,-1., 0.], [0.,-1., 1.]]
        dir /= N.sqrt(N.sum(dir**2, axis=0))
        
        self.bund = RayBundle(pos, dir)
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

class TestFiniteCylinder(unittest.TestCase):
    """Just see that it throws out the right rays"""
    def setUp(self):
        # Two rays inside, two outside; two horizontal, two slanted.
        pos = N.c_[[0., 0., 0.], [0., 0., 0.], [0., 1., 0.], [0., 1., 0.],
            [0., 1., 0.], [0., 1., 0.1], [0., 0.2, -0.06], [0., 0., 0.03]]
        dir = N.c_[[0., 1., 0.], [0., 1., 1.], [0.,-1., 0.], [0.,-1., 1.],
            [0., -1., 0.05], [0., -1., -0.05], [0., 1., 0.2], [0., 1., 0.1]]
        dir /= N.sqrt(N.sum(dir**2, axis=0))
        
        self.bund = RayBundle(pos, dir)
        self.gm = FiniteCylinder(diameter=1., height=0.1)
    
    def test_vertical(self):
        """Finite cylinder parallel to the Z axis"""
        correct_prm = N.r_[0.5, 0, 0.5, 0, 0.50062461, 1.50187383, 0.30594117, 0.]
        prm = self.gm.find_intersections(N.eye(4), self.bund)
        N.testing.assert_array_equal(prm[[1,3, 7]], N.ones(3)*N.inf)
        prm[[1,3, 7]] = 0.
        N.testing.assert_array_almost_equal(prm, correct_prm)
        
        self.gm.select_rays(N.r_[0, 2])
        
        correct_norm = N.c_[[0.,-1., 0.], [0.,1., 0.]]
        norm = self.gm.get_normals()
        N.testing.assert_array_almost_equal(norm, correct_norm)
        
        correct_pts = N.c_[[0., 0.5, 0.], [0., 0.5, 0.]]
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_almost_equal(pts, correct_pts)
    
    def test_rotated(self):
        """Rotate finite cylinder by 30 degrees"""
        frame = rotx(N.pi/6)
        
        correct_prm = N.empty(8)
        correct_prm.fill(N.inf)
        prm = self.gm.find_intersections(frame, self.bund)
        N.testing.assert_array_equal(prm, correct_prm)
    
    def test_mesh(self):
        """Cylindrical mesh OK"""
        x, y, z = self.gm.mesh(20)
        self.failUnless(N.allclose(x**2 + y**2, 0.25))


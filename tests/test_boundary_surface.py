# Unit tests for the boundary shape class and the boundary sphere class.

import unittest
import numpy as N

from tracer.spatial_geometry import generate_transform
from tracer.boundary_shape import *

class TestSphereBoundingRect(unittest.TestCase):
    def setUp(self):
        # Create some surfaces to intersect with spheres.
        self.at_origin_xy = N.eye(4)
        self.at_origin_yz = generate_transform(N.r_[0, 1, 0], N.pi/2, \
            N.c_[[0,0,0]])
        self.at_origin_slant = generate_transform(N.r_[0, 1, 0], N.pi/4, \
            N.c_[[0,0,0]])
        
        self.parallel_xy = generate_transform(N.r_[1, 0, 0], 0, N.c_[[0, 0, 1]])
        self.parallel_yz = self.at_origin_yz.copy()
        self.parallel_yz[0,3] += 1
        self.parallel_slanted = self.at_origin_slant.copy()
        self.parallel_slanted[[0,2],3] += N.sqrt(0.5)
    
    def test_sphere_at_origin(self):
        """For a bounding sphere at the origin, the right bounding rects are returned"""
        sphere = BoundarySphere(radius=2.)
        
        extents = sphere.bounding_rect_for_plane(self.at_origin_xy)
        self.failUnlessEqual(extents, (-2., 2., -2., 2.))
        
        extents = sphere.bounding_rect_for_plane(self.at_origin_yz)
        self.failUnlessEqual(extents, (-2., 2., -2., 2.))
        
        extents = sphere.bounding_rect_for_plane(self.at_origin_slant)
        self.failUnlessEqual(extents, (-2., 2., -2., 2.))
        
        sqrt_3 = N.sqrt(3)
        extents = sphere.bounding_rect_for_plane(self.parallel_xy)
        self.failUnlessEqual(extents, (-sqrt_3, sqrt_3, -sqrt_3, sqrt_3))
        
        extents = sphere.bounding_rect_for_plane(self.parallel_yz)
        self.failUnlessEqual(extents, (-sqrt_3, sqrt_3, -sqrt_3, sqrt_3))
        
        extents = sphere.bounding_rect_for_plane(self.parallel_slanted)
        N.testing.assert_array_almost_equal(extents, \
            (-sqrt_3, sqrt_3, -sqrt_3, sqrt_3))
    
    def test_sphere_moved(self):
        """For a bounding sphere at 1,0,0 the right bounding rects are returned"""
        sphere = BoundarySphere(radius=2., location=N.r_[1,0,0])
        
        extents = sphere.bounding_rect_for_plane(self.at_origin_xy)
        self.failUnlessEqual(extents, (-1., 3., -2., 2.))

        sqrt_3 = N.sqrt(3)
        extents = sphere.bounding_rect_for_plane(self.at_origin_yz)
        self.failUnlessEqual(extents, (-sqrt_3, sqrt_3, -sqrt_3, sqrt_3))

        sqrt_h = N.sqrt(0.5)
        sqrt_35 = N.sqrt(3.5)
        extents = sphere.bounding_rect_for_plane(self.at_origin_slant)
        self.failUnlessEqual(extents, \
            (sqrt_h - sqrt_35, sqrt_h + sqrt_35, -sqrt_35, sqrt_35))

        extents = sphere.bounding_rect_for_plane(self.parallel_xy)
        self.failUnlessEqual(extents, (1 - sqrt_3, 1 + sqrt_3, -sqrt_3, sqrt_3))

        extents = sphere.bounding_rect_for_plane(self.parallel_yz)
        self.failUnlessEqual(extents, (-2., 2., -2., 2.))

        Reff = N.sqrt(4 - (1 - N.sqrt(0.5))**2)
        extents = sphere.bounding_rect_for_plane(self.parallel_slanted)
        N.testing.assert_array_almost_equal(extents, \
            (sqrt_h - Reff, sqrt_h + Reff, -Reff, Reff))

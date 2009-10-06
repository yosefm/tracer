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

        self.gm = FlatGeometryManager()
        self.prm = self.gm.find_intersections(N.eye(4), self._bund)
        
    def test_find_intersections(self):
        """The correct parametric locations are found for flat geometry"""
        self.failUnlessEqual(self.prm.shape, (4,), 
            "Shape of parametric location array is wrong: " + \
            str(self.prm.shape))
        N.testing.assert_array_almost_equal(self.prm, N.sqrt(3))
    
    def test_get_normals(self):
        """A flat geometry manager returns parallel normals"""
        n = self.gm.get_normals(N.ones(4, dtype=N.bool)) # all rays selected
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, 1]], (1,4)))
        
        n = self.gm.get_normals(N.array([0, 1, 0, 1], dtype=N.bool)) # just two rays selected
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, 1]], (1,2)))
    
    def test_inters_points_global(self):
        """On the basic setup, a flat surface returns correct intersections"""
        correct_pts = N.zeros((3,4))
        correct_pts[:2,0] = 1
        
        pts = self.gm.get_intersection_points_global(N.ones(4, dtype=N.bool))
        N.testing.assert_array_equal(pts, correct_pts)
        
        pts = self.gm.get_intersection_points_global(N.array([0, 1, 0, 1], dtype=N.bool))
        N.testing.assert_array_equal(pts, correct_pts[:,[1,3]])

import spatial_geometry as SP
class TestFlatGeomTilted(unittest.TestCase):
    """Use a flat surface rotated about the x axis by 45 degrees"""
    def setUp(self):
        s2 = math.sqrt(2)
        dir = N.c_[[1, 0, -s2], [-1, 0, -s2], [-1, -s2, 0], [1, -s2, 0]] / math.sqrt(3)
        position = N.c_[[0,1/s2,1/s2], [1,0,s2], [1,s2,0], [-1,s2,0]]
        
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)

        self.gm = FlatGeometryManager()
        frame = SP.generate_transform(N.r_[1., 0, 0], -N.pi/4., N.zeros((3,1)))
        self.prm = self.gm.find_intersections(frame, self._bund)
        
    def test_find_intersections(self):
        """The correct parametric locations are found for flat geometry"""
        self.failUnlessEqual(self.prm.shape, (4,), 
            "Shape of parametric location array is wrong: " + \
            str(self.prm.shape))
        N.testing.assert_array_almost_equal(self.prm, math.sqrt(3))
    
    def test_get_normals(self):
        """A flat geometry manager returns parallel normals"""
        s2 = math.sqrt(2)
        n = self.gm.get_normals(N.ones(4, dtype=N.bool)) # all rays selected
        N.testing.assert_array_almost_equal(n, N.tile(N.c_[[0,1/s2,1/s2]], (1,4)))
        
        n = self.gm.get_normals(N.array([0, 1, 0, 1], dtype=N.bool)) # just two rays selected
        N.testing.assert_array_almost_equal(n, N.tile(N.c_[[0,1/s2,1/s2]], (1,2)))
    
    def test_inters_points_global(self):
        """On the basic setup, a flat surface returns correct intersections"""
        correct_pts = N.zeros((3,4))
        s2 = math.sqrt(2)
        correct_pts[:,0] = N.r_[1, 1/s2, -1/s2]
                
        pts = self.gm.get_intersection_points_global(N.ones(4, dtype=N.bool))
        N.testing.assert_array_almost_equal(pts, correct_pts)
        
        pts = self.gm.get_intersection_points_global(N.array([0, 1, 0, 1], dtype=N.bool))
        N.testing.assert_array_almost_equal(pts, correct_pts[:,[1,3]])


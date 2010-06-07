# Test the flat geometry manager for conformance to the required interface

import unittest
import numpy as N
import math

from tracer.flat_surface import FlatGeometryManager
from tracer.ray_bundle import RayBundle

class TestFlatGeomManagerInterface(unittest.TestCase):
    def setUp(self):
        dir = N.c_[[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]] / math.sqrt(3)
        position = N.c_[[0,0,1], [1,-1,1], [1,1,1], [-1,1,1]]
        self._bund = RayBundle(position, dir)

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
        self.gm.select_rays(N.arange(4))
        n = self.gm.get_normals()
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, 1]], (1,4)))
        
    def test_select_rays_normals(self):
        """Correct normals when some rays not selected"""
        self.gm.select_rays(N.r_[1,3])
        n = self.gm.get_normals()
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, 1]], (1,2)))
    
    def test_inters_points_global(self):
        """On the basic setup, a flat surface returns correct intersections"""
        correct_pts = N.zeros((3,4))
        correct_pts[:2,0] = 1
        
        self.gm.select_rays(N.arange(4))
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_equal(pts, correct_pts)
        
    def select_rays_inters(self):
        """Correct intersections when some rays not selected"""
        correct_pts = N.zeros((3,2))
        correct_pts[:2,0] = 1
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_equal(pts, correct_pts)

import tracer.spatial_geometry as SP
class TestFlatGeomTilted(unittest.TestCase):
    """Use a flat surface rotated about the x axis by 45 degrees"""
    def setUp(self):
        s2 = math.sqrt(2)
        dir = N.c_[[1, 0, -s2], [-1, 0, -s2], [-1, -s2, 0], [1, -s2, 0]] / math.sqrt(3)
        position = N.c_[[0,1/s2,1/s2], [1,0,s2], [1,s2,0], [-1,s2,0]]
        self._bund = RayBundle(position, dir)

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
        """A tilted flat geometry manager returns parallel normals"""
        s2 = math.sqrt(2)
        self.gm.select_rays(N.arange(4))
        n = self.gm.get_normals()
        N.testing.assert_array_almost_equal(n, N.tile(N.c_[[0,1/s2,1/s2]], (1,4)))
        
    def test_select_rays_normals(self):
        """A tilted flat geometry manager returns normals only for selected rays"""
        s2 = math.sqrt(2)
        self.gm.select_rays(N.r_[1,3])
        n = self.gm.get_normals()
        N.testing.assert_array_almost_equal(n, N.tile(N.c_[[0,1/s2,1/s2]], (1,2)))
    
    def test_inters_points_global(self):
        """On the basic setup, a tilted flat surface returns correct intersections"""
        correct_pts = N.zeros((3,4))
        s2 = math.sqrt(2)
        correct_pts[:,0] = N.r_[1, 1/s2, -1/s2]
        
        self.gm.select_rays(N.arange(4))
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_almost_equal(pts, correct_pts)
    
    def test_select_rays_inters(self):
        """With dropped rays, a tilted flat surface returns correct intersections"""
        s2 = math.sqrt(2)
        correct_pts = N.zeros((3,2))
        
        self.gm.select_rays(N.r_[1,3])
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_almost_equal(pts, correct_pts)

class TestFlatGeomTranslated(unittest.TestCase):
    def setUp(self):
        dir = N.c_[[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]] / math.sqrt(3)
        position = N.c_[[0,0,1], [1,-1,1], [1,1,1], [-1,1,1]]
        self._bund = RayBundle(position, dir)
        
        self.gm = FlatGeometryManager()
        frame = SP.translate(1., 0., 0.)
        self.prm = self.gm.find_intersections(frame, self._bund)
        
    def test_find_intersections(self):
        """The correct parametric locations are found for translated flat geometry"""
        self.failUnlessEqual(self.prm.shape, (4,),
            "Shape of parametric location array is wrong: " + \
            str(self.prm.shape))
        N.testing.assert_array_almost_equal(self.prm, N.sqrt(3))
    
    def test_get_normals(self):
        """A translated flat geometry manager returns parallel normals"""
        self.gm.select_rays(N.arange(4))
        n = self.gm.get_normals()
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, 1]], (1,4)))
    
    def test_inters_points_global(self):
        """When translated, a flat surface returns correct intersections"""
        correct_pts = N.zeros((3,4))
        correct_pts[:2,0] = 1
        
        self.gm.select_rays(N.arange(4))
        pts = self.gm.get_intersection_points_global()
        N.testing.assert_array_equal(pts, correct_pts)

class TestBacksideNormals(unittest.TestCase):
    def setUp(self):
        dir = N.c_[[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]] / math.sqrt(3)
        position = N.c_[[0,0,-1], [1,-1,-1], [1,1,-1], [-1,1,-1]]
        self._bund = RayBundle(position, dir)

        self.gm = FlatGeometryManager()
        self.prm = self.gm.find_intersections(N.eye(4), self._bund)
    
    def test_get_normals(self):
        """A translated flat geometry manager returns parallel normals"""
        self.gm.select_rays(N.arange(4))
        n = self.gm.get_normals()
        N.testing.assert_array_equal(n, N.tile(N.c_[[0, 0, -1]], (1,4)))


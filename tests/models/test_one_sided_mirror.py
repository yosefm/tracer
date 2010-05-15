from tracer.models.one_sided_mirror import *
from tracer.ray_bundle import RayBundle
from tracer.tracer_engine import TracerEngine
from tracer.assembly import Assembly
import tracer.spatial_geometry as sp

import unittest
import numpy as N

class TestRectOneSided(unittest.TestCase):
    def setUp(self):
        self.mirror = rect_one_sided_mirror(1.5, 1.5, 0.9)
        
        pos = N.zeros((3,8))
        pos[0] = N.tile(N.r_[0, 0.5, 2, -2], 2)
        pos[2] = N.repeat(N.r_[1, -1], 4)
        dir = N.zeros((3,8))
        dir[2] = N.repeat(N.r_[-1, 1], 4)
        
        self.bund = RayBundle()
        self.bund.set_vertices(pos)
        self.bund.set_directions(dir)
        self.bund.set_energy(N.ones(8)*1000)
        self.bund.set_ref_index(N.ones(8))
    
    def test_regular(self):
        """One-sided plate without rotation"""
        e = TracerEngine(Assembly(objects=[self.mirror]))
        e.ray_tracer(self.bund, 1, 0.05)
        outg = e.tree[-1]
        
        correct_verts = N.zeros((3,2))
        correct_verts[0] = N.r_[0, 0.5]
        N.testing.assert_array_equal(
            outg.get_vertices()[:,outg.get_energy() > 0], correct_verts)
        N.testing.assert_array_almost_equal(
            outg.get_energy(), N.r_[100., 100., 0, 0])
    
    def test_rotated(self):
        """One-sided plate with rotation"""
        rot = sp.roty(N.pi/4.)
        self.mirror.set_transform(rot)
        
        e = TracerEngine(Assembly(objects=[self.mirror]))
        e.ray_tracer(self.bund, 1, 0.05)
        outg = e.tree[-1]
        
        correct_verts = N.array([[0., 0.5], [0., 0.], [0., -0.5]])
        N.testing.assert_array_almost_equal(
            outg.get_vertices()[:,outg.get_energy() > 0], correct_verts)
        N.testing.assert_array_almost_equal(
            outg.get_energy(), N.r_[100., 100., 0, 0])

# -*- coding: utf-8 -*-
# Unit tests for the mini-dish model.

import unittest
import numpy as N

from tracer.models.tau_minidish import MiniDish
from tracer.spatial_geometry import roty
import tracer.tracer_engine

class TestMiniDish(unittest.TestCase):
    def setUp(self):
        """Create a basic minidish assembly to abuse"""
        self.md = MiniDish(5, 5, 0.9, 5.7, .4, 0.7, 0.9)
        self.e = tracer.tracer_engine.TracerEngine(self.md)
        
        pos = N.zeros((3, 5))
        pos[0] = N.r_[-2:2:5j]
        pos[2] = 6.
        self.pos = pos

        dir = N.zeros((3, 5))
        dir[2] = -1.
        self.dir = dir
        
    def test_upright(self):
        """Dish without rotation"""
        bund = tracer.ray_bundle.RayBundle()
        
        bund.set_vertices(self.pos)
        bund.set_directions(self.dir)
        
        bund.set_energy(N.ones(5)*100)
        bund.set_ref_index(N.ones(5))
        
        self.e.ray_tracer(bund, 1776, 0.05)
        
        receiver = self.md.get_receiver_surf()
        energy, pts = receiver.get_optics_manager().get_all_hits()
        x, y = receiver.global_to_local(pts)[:2]
        
        self.failUnless(N.allclose(y, 0))
        N.testing.assert_array_equal(energy, N.r_[90., 90., 81., 81.])
    
    def test_rotated(self):
        """Dish with rotation"""
        bund = tracer.ray_bundle.RayBundle()
        
        rot = roty(N.pi/4)
        self.md.set_transform(rot)
        bund.set_vertices(N.dot(rot[:3,:3], self.pos))
        bund.set_directions(N.dot(rot[:3,:3], self.dir))
        
        bund.set_energy(N.ones(5)*100)
        bund.set_ref_index(N.ones(5))
        
        self.e.ray_tracer(bund, 1776, 0.05)
        
        receiver = self.md.get_receiver_surf()
        energy, pts = receiver.get_optics_manager().get_all_hits()
        x, y = receiver.global_to_local(pts)[:2]
        
        self.failUnless(N.allclose(y, 0))
        N.testing.assert_array_equal(energy, N.r_[90., 90., 81., 81.])

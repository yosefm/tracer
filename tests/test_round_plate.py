# -*- coding: utf-8 -*-
# Tests for the round plate geometry manager.

import unittest
import numpy as N

from tracer.flat_surface import RoundPlateGM
from tracer.ray_bundle import RayBundle
from tracer.surface import Surface
import tracer.optics_callables as opt

class TestRectPlateGM(unittest.TestCase):
    def test_value_error(self):
        """Can't create a negative round plate"""
        self.assertRaises(ValueError, RoundPlateGM, -1)
    
    def test_selection(self):
        pos = N.zeros((3,4))
        pos[0] = N.r_[0, 0.5, 2, -2]
        pos[2] = 1.
        dir = N.tile(N.c_[[0,0,-1]], (1,4))
        
        bund = RayBundle()
        bund.set_vertices(pos)
        bund.set_directions(dir)
        
        surf = Surface(RoundPlateGM(1), opt.perfect_mirror)
        misses = N.isinf(surf.register_incoming(bund))
        
        N.testing.assert_array_equal(misses, N.r_[False, False, True, True])
        
    def test_mesh(self):
        """The mesh representing the circle looks right"""
        p = RoundPlateGM(5)
        x, y, z = p.mesh(5)
        
        N.testing.assert_array_equal(z, 0) # Easy
        self.failUnless(N.any(x**2 + y**2 > 25))


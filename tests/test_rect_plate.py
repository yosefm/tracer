# Tests for the rectangular plate geometry manager.

import unittest
import numpy as N

from tracer.flat_surface import RectPlateGM
from tracer.ray_bundle import RayBundle
from tracer.surface import Surface
import tracer.optics_callables as opt

class TestRectPlateGM(unittest.TestCase):
    def test_value_error(self):
        """Can't create a negative rect-plate"""
        self.assertRaises(ValueError, RectPlateGM, -1, 7)
        self.assertRaises(ValueError, RectPlateGM, 1, -7)
    
    def test_selection(self):
        pos = N.zeros((3,4))
        pos[0] = N.r_[0, 0.5, 2, -2]
        pos[2] = 1.
        dir = N.tile(N.c_[[0,0,-1]], (1,4))
        
        bund = RayBundle()
        bund.set_vertices(pos)
        bund.set_directions(dir)
        
        surf = Surface(RectPlateGM(1, 0.25), opt.perfect_mirror)
        misses = N.isinf(surf.register_incoming(bund))
        
        N.testing.assert_array_equal(misses, N.r_[False, False, True, True])
    
    def test_mesh(self):
        """Correct mesh for recxt-plate"""
        r = RectPlateGM(5, 6)
        res = 0.1
        x, y, z = r.mesh(res)
        
        cx, cy = N.mgrid[-2.5:2.51:res, -3:3.01:0.1]
        N.testing.assert_array_equal(x, cx)
        N.testing.assert_array_equal(y, cy)
        N.testing.assert_array_equal(z, N.zeros_like(x))

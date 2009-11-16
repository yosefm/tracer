# Tests for the rectangular plate geometry manager.

import unittest
import numpy as N

from flat_surface import RectPlateGM
from ray_bundle import RayBundle
from surface import Surface
import optics_callables as opt

class TestRectPlateGM(unittest.TestCase):
    def test_value_error(self):
        """Can't create a negative rect-plate"""
        self.assertRaises(ValueError, RectPlateGM, -1, 7)
        self.assertRaises(ValueError, RectPlateGM, 1, -7)
    
    def test_selection(self):
        pos = N.zeros((3,4))
        pos[0] = N.r_[0, 0.5, 2, -2]
        dir = N.tile(N.c_[[0,0,-1]], (1,4))
        
        bund = RayBundle()
        bund.set_vertices(pos)
        bund.set_directions(dir)
        
        surf = Surface(RectPlateGM(1, 0.25), opt.perfect_mirror)
        misses = N.isinf(surf.register_incoming(bund))
        
        N.testing.assert_array_equal(misses, N.r_[False, False, True, True])


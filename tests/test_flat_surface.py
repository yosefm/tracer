import unittest
import math
import numpy as N
from numpy import c_, r_

from tracer.surface import Surface
from tracer.flat_surface import FlatGeometryManager
from tracer.optics_callables import gen_reflective
from tracer.ray_bundle import RayBundle

class TestTraceProtocol(unittest.TestCase):
    def setUp(self):
        self._surf = Surface(FlatGeometryManager(), gen_reflective(0))
        
        dir = N.array([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]).T / math.sqrt(3)
        position = c_[[0,0,1], [1,-1,1], [1,1,1], [-1,1,1]]
        
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.ones(4)*100)
        self._bund.set_ref_index(N.r_[1, 1, 1, 1])
    
    def test_register_incoming(self):
        """Flat mirror: a simple bundle is registered correctly"""
        correct_params = r_[[math.sqrt(3)]*4]
        params = self._surf.register_incoming(self._bund)
        
        N.testing.assert_array_almost_equal(params, correct_params)
    
    def test_get_outgoing(self):
        """Flat mirror: the correct outgoing bundle is returned"""
        params = self._surf.register_incoming(self._bund)
        outg = self._surf.get_outgoing(N.ones(4, dtype=N.bool))
        
        correct_pts = N.zeros((3,4))
        correct_pts[:2,0] = 1
        N.testing.assert_array_equal(outg.get_vertices(), correct_pts)
        
        correct_dirs = N.c_[[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]] / N.sqrt(3)
        N.testing.assert_array_equal(outg.get_directions(), correct_dirs)
        
        N.testing.assert_array_equal(outg.get_energy(), N.ones(4)*100)
        N.testing.assert_array_equal(outg.get_parent(), N.arange(4))


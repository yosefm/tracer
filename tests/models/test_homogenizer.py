
import unittest
import numpy as N

from tracer.models.homogenizer import rect_homogenizer
from tracer.ray_bundle import RayBundle
from tracer.tracer_engine import TracerEngine

class TestHomogenizer(unittest.TestCase):
    def setUp(self):
        """A homogenizer transforms a bundle correctly"""
        hmg = rect_homogenizer(5., 3., 10., 0.9)
        self.engine = TracerEngine(hmg)
        self.bund = RayBundle()
        
        # 4 rays starting somewhat above (+z) the homogenizer
        pos = N.zeros((3,4))
        pos[2] = N.r_[11, 11, 11, 11]
        self.bund.set_vertices(pos)
        
        # One ray going to each wall:
        dir = N.c_[[1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1]]/N.sqrt(2)
        self.bund.set_directions(dir)
        
        # Laborious setup details:
        self.bund.set_energy(N.ones(4)*4.)
        self.bund.set_ref_index(N.ones(4))
    
    def test_first_hits(self):
        """Test bundle enters homogenizer correctly"""
        v, d = self.engine.ray_tracer(self.bund, 1, 0.05)
        
        out_dirs = N.c_[[-1, 0, -1], [1, 0, -1], [0, -1, -1], [0, 1, -1]]/N.sqrt(2)
        N.testing.assert_array_almost_equal(d, out_dirs)
        
        out_hits = N.c_[
            [2.5, 0, 8.5], 
            [-2.5, 0, 8.5], 
            [0, 1.5, 9.5], 
            [0, -1.5, 9.5]]
        N.testing.assert_array_almost_equal(v, out_hits)
    

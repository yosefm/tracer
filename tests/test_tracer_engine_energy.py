import unittest
import numpy as N
import math

from tracer_engine import TracerEngine
import ray_bundle
from flat_surface import FlatSurface
from spatial_geometry import general_axis_rotation
from receiver import Receiver
import pdb

class TestEnergyProtocol(unittest.TestCase):
    """
    Tests the energy plot
    """
    def setUp(self):

        self.x = 1/(math.sqrt(2))
        dir = N.array([0,-self.x,self.x])
        position = N.array([0,2,1]).reshape(-1,1)

        self.bund = ray_bundle.solar_disk_bundle(5000, position, dir, 1.5, N.pi/1000.)

        rot1 = general_axis_rotation([1,0,0],N.pi/4)
        energy = N.ones(5000)
        self.bund.set_energy(energy)
        self.objects = [Receiver(rotation=rot1,width=10,height=10), 
                   FlatSurface(width=10,height=10)]
        self.engine = TracerEngine(self.objects)
        
    def test_ray_tracer1(self):
        self.engine.ray_tracer(self.bund, 1)
        self.objects[0].plot_energy()


"""
    def test_ray_tracer2(self):
        params = self.engine.ray_tracer(self._bund, 2)
        correct_params = N.c_[[0,2,2],[0,3,0]]

        N.testing.assert_array_almost_equal(params,correct_params)
"""
if __name__ == '__main__':
    unittest.main()

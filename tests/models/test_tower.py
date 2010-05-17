"""
Test solar-tower components.
"""

import unittest
import numpy as N

from tracer.tracer_engine import TracerEngine
from tracer.ray_bundle import RayBundle
from tracer.models.heliostat_field import HeliostatField, radial_stagger

class TestHeliostatField(unittest.TestCase):
    def setUp(self):
        spread = N.r_[50:101:10]
        self.pos = N.zeros((2*len(spread), 3))
        self.pos[:len(spread), 0] = spread
        self.pos[len(spread):, 1] = spread
        self.pos[:,2] = 4.5
        
        self.field = HeliostatField(self.pos, 8., 8., 0., 90.)
        s2 = N.sqrt(2)/2
        self.sunvec = N.r_[-s2, 0, s2] # Noon, winterish.
        
        ray_pos = (self.pos + self.sunvec).T
        ray_dir = N.tile(-self.sunvec, (self.pos.shape[0], 1)).T
        self.rays = RayBundle(ray_pos, ray_dir, energy=N.ones(self.pos.shape[0]))
    
    def test_secure_position(self):
        """Heliostats at default position absorb the sunlight"""
        e = TracerEngine(self.field)
        e.ray_tracer(self.rays, 1, 0.05)
        
        N.testing.assert_array_equal(e.tree[-1].get_energy(), 0)
    
    def test_aim(self):
        """Aiming heliostats works"""
        elev = N.pi/4
        az = N.pi/2
        self.field.aim_to_sun(az, elev)
        
        e = TracerEngine(self.field)
        v, d = e.ray_tracer(self.rays, 1, 0.05)
        
        N.testing.assert_array_almost_equal(d[1, :self.pos.shape[0]/2], 0)
        N.testing.assert_array_almost_equal(d[0, self.pos.shape[0]/2:], 0)
        N.testing.assert_array_almost_equal(abs(d[2]*(v[0] + v[1])/(d[0] + d[1])), 85.5)

class TestRadialStagger(unittest.TestCase):
    def test_stagger(self):
        """position in a radial-stagger are correct"""
        pos = radial_stagger(-N.pi/4, N.pi/4 + 0.0001, N.pi/2, 5, 10, 1)
        
        N.testing.assert_array_almost_equal(
            N.sqrt(N.sum(pos**2, axis=1)), 
            N.r_[5, 5, 7, 7, 9, 9, 6, 8])


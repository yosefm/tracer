# Test suit for the ray_bundle module

import math
import unittest
import numpy as N
from scipy import stats
import tracer.ray_bundle as RB

class TestDistributions(unittest.TestCase):
    def assert_radius(self,  vertices,  center,  R):
        """Asserts that the vertices of a bundle are within the given radius of
        the bundle's center.
        Arguments: vertices - as returned from RayBundle.get_vertices()
            center - the center used to construct the bundle.
            R - the radius of the disk containing the vertices.
        """
        self.assert_((((vertices - center)**2).sum(axis=0) <= R**2).all())
    
    def assert_uniform(self, sample, lb=0, ub=1, confidence=0.05):
        """Asserts that the given sample is distributed uniformly, with values
        between lb and hb, with given confidence level . The null hypothesis is that
        the distribution is the given uniform distribution.
        Arguments: sample - a 1D vector of sample points.
            lb, ub - the lower and upper bound, respectively,  of the uniform 
                distribution tested.
            confidence - the null hypothesis is accepted if the Kolmogorov-Smirnov 
                test gives a p-value > confidence.
        """
        normed = (sample - lb)/(ub - lb)
        D, p = stats.kstest(normed, 'uniform')
        self.assert_(p > confidence, "%g too low" % p)
        
    def test_location(self):
        dir = N.array([0., 0, 1])
        center = N.array([0,  0, 0]).reshape(-1, 1) 
        R = 2
        
        rays = RB.solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
        
        center = N.array([7,  7,  7]).reshape(-1, 1) 
        rays = RB.solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
        
    def test_rotation(self):
        dir = N.array([0., 0, 1])
        center = N.array([0,  0, 0]).reshape(-1, 1) 
        R = 2
        
        rays = RB.solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
        
        dir = 1/math.sqrt(3)*N.ones(3)
        rays = RB.solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
    
    def test_ray_directions(self):
        """Test that the angle between the rays and the bundle direction is uniformly
        spaced, and that the rays are rotated around the bundle direction at uniformly
        distributed angles.
        """
        dir = N.array([0., 0, 1])
        center = N.array([0,  0, 0]).reshape(-1, 1) 
        R = 2; theta_max = N.pi/100.
        
        rays = RB.solar_disk_bundle(5000, center, dir, R, theta_max)
        directs = rays.get_directions()
        dir_dots = N.dot(dir, directs)
        self.assert_((dir_dots >= N.cos(theta_max)).all())
        self.assert_uniform(N.arccos(dir_dots), ub=theta_max)
        
        on_disk_plane = directs - N.outer(dir, dir_dots)
        angs = N.arctan2(on_disk_plane[0], on_disk_plane[1])
        self.assert_uniform(angs, lb=-N.pi, ub=N.pi)
        
if __name__ == '__main__':
    unittest.main()

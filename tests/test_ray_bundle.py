# Test suit for the ray_bundle module

import math
import unittest

import numpy as N
from scipy import stats

import tracer.ray_bundle as RB
from tracer.sources import solar_disk_bundle

class TestInheritance(unittest.TestCase):
    def test_inherit_empty_from_empty(self):
        """Inherit empty bundle completely"""
        father = RB.RayBundle()
        child = father.inherit()
    
    def test_inherit_empty_from_full(self):
        """Inherit a populated bundle completely"""
        pos = N.ones((3,4))
        dir = N.zeros((3,4))
        energy = N.ones(4)
        ref_ind = N.zeros(4)
        prn = N.ones(4)
        
        father = RB.RayBundle(pos, dir, energy, prn, ref_ind)
        child = father.inherit()
        
        N.testing.assert_array_equal(child.get_vertices(), pos)
        N.testing.assert_array_equal(child.get_directions(), dir)
        N.testing.assert_array_equal(child.get_energy(), energy)
        N.testing.assert_array_equal(child.get_ref_index(), ref_ind)
        N.testing.assert_array_equal(child.get_parents(), prn)
    
    def test_inherit_full_from_full(self):
        """Inherit a populated bundle from otherwise-populated bundle"""
        pos = N.ones((3,4))
        dir = N.zeros((3,4))
        energy = N.ones(4)
        ref_ind = N.zeros(4)
        prn = N.ones(4)

        father = RB.RayBundle(pos, dir, energy, prn, ref_ind)
        child = father.inherit(N.s_[:], dir, pos, prn, energy, N.ones(4)*5)
        
        N.testing.assert_array_equal(child.get_vertices(), dir)
        N.testing.assert_array_equal(child.get_directions(), pos)
        N.testing.assert_array_equal(child.get_energy(), prn)
        N.testing.assert_array_equal(child.get_ref_index(), N.ones(4)*5)
        N.testing.assert_array_equal(child.get_parents(), energy)
    
    def test_inherit_part_from_full(self):
        """Inherit part of a populated bundle"""
        pos = N.ones((3,4))
        dir = N.zeros((3,4))
        
        father = RB.RayBundle(pos, dir)
        child = father.inherit(selector=N.r_[0,1], direction=N.ones((3,2)))
        
        N.testing.assert_array_equal(child.get_vertices(), pos[:,[0,1]])
        N.testing.assert_array_equal(child.get_directions(), N.ones((3,2)))
    
    def test_delete_rays(self):
        """delete_rays() does its job"""
        pos = N.ones((3,4))
        direct = N.zeros((3,4))
        prn = N.arange(4)
        
        father = RB.RayBundle(pos, direct, parents=prn)
        child = father.delete_rays(N.r_[0])
        
        N.testing.assert_array_equal(child.get_vertices(), N.ones((3,3)))
        N.testing.assert_array_equal(child.get_directions(), N.zeros((3,3)))
        N.testing.assert_array_equal(child.get_parents(), N.arange(1,4))
    
    def test_extra_property(self):
        pos = N.ones((3,4))
        direct = N.zeros((3,4))
        wavelength = N.arange(4)
        
        father = RB.RayBundle(pos, direct, wavelength=wavelength)
        child = father.inherit()
        
        N.testing.assert_array_equal(child.get_wavelength(), wavelength)

class TestConcatenate(unittest.TestCase):
    def test_concat(self):
        r1 = RB.RayBundle(N.ones((3,4)), N.ones((3,4)))
        r2 = RB.RayBundle(N.zeros((3,4)), N.zeros((3,4)))
        con = RB.concatenate_rays((r1, r2))
        correct = N.hstack((N.ones((3,4)), N.zeros((3,4)) ))
        N.testing.assert_array_equal(con.get_vertices(), correct)
        N.testing.assert_array_equal(con.get_directions(), correct)
    
    def test_extra_prop(self):
        r1 = RB.RayBundle(N.ones((3,4)), N.ones((3,4)), wavelen=N.ones((3,4)))
        r2 = RB.RayBundle(N.zeros((3,4)), N.zeros((3,4)), wavelen=N.zeros((3,4)))
        con = RB.concatenate_rays((r1, r2))
        correct = N.hstack((N.ones((3,4)), N.zeros((3,4)) ))
        N.testing.assert_array_equal(con.get_wavelen(), correct)
        
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
        
        rays = solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
        
        center = N.array([7,  7,  7]).reshape(-1, 1) 
        rays = solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
        
    def test_rotation(self):
        dir = N.array([0., 0, 1])
        center = N.array([0,  0, 0]).reshape(-1, 1) 
        R = 2
        
        rays = solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
        
        dir = 1/math.sqrt(3)*N.ones(3)
        rays = solar_disk_bundle(1000,  center,  dir,  R,  N.pi/100.)
        self.assert_radius(rays.get_vertices(),  center,  R)
    
    def test_ray_directions(self):
        """Test that the angle between the rays and the bundle direction is uniformly
        spaced, and that the rays are rotated around the bundle direction at uniformly
        distributed angles.
        """
        dir = N.array([0., 0, 1])
        center = N.array([0,  0, 0]).reshape(-1, 1) 
        R = 2; theta_max = N.pi/100.
        
        rays = solar_disk_bundle(5000, center, dir, R, theta_max)
        directs = rays.get_directions()
        dir_dots = N.dot(dir, directs)
        self.failUnless((dir_dots >= N.cos(theta_max)).all())
        
        on_disk_plane = directs - N.outer(dir, dir_dots)
        angs = N.arctan2(on_disk_plane[0], on_disk_plane[1])
        self.assert_uniform(angs, lb=-N.pi, ub=N.pi)
    
    def test_directions_rotated(self):
        """Ray directions OK when the bundle directions isn't on one of the axes"""
        dir = N.array([0., N.sqrt(2), N.sqrt(2)])/2.
        center = N.c_[[0,  0, 0]]
        R = 2; theta_max = N.pi/100.
        
        rays = solar_disk_bundle(5000, center, dir, R, theta_max)
        directs = rays.get_directions()
        dir_dots = N.dot(dir, directs)
        self.failUnless((dir_dots >= N.cos(theta_max)).all())
    
    def test_energy(self):
        """When flux is given, the solar disk bundle has proper energy"""
        dir = N.array([0., 0, 1])
        center = N.array([0,  0, 0]).reshape(-1, 1)
        R = 2; theta_max = N.pi/100.

        rays = solar_disk_bundle(5000, center, dir, R, theta_max, flux=1000.)
        N.testing.assert_array_equal(rays.get_energy(), N.pi*4/5.)
        
if __name__ == '__main__':
    unittest.main()

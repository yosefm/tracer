# Unit tests for the optics module.
#
# Reference:
# [1] Grant R. Fowles, Introduction to Modern Optics, 2nd ed.; Dover 
#     Publications Inc.; 1989

import math
import optics
import unittest
import numpy as N
from ray_bundle import RayBundle

class TestSingleReflection(unittest.TestCase):
    def runTest(self):
        """A single beam at 45 degs to the surface reflects correctly"""
        dir = N.array([[0, 1, -1]]).T / math.sqrt(2)
        normal = N.array([[0, 0, 1]]).T
        correct_reflection = N.array([[0, 1, 1]]).T / math.sqrt(2)
        
        reflection = optics.reflections(dir, normal)
        self.failUnless((reflection == correct_reflection).all(), 
            "Reflection is\n" + str(reflection) + "\nbut should be\n" + str(correct_reflection))

class TestMultipleReflections(unittest.TestCase):
    """Four symmetric rays reflect to the correct directions"""
    def runTest(self):
        dir = N.array([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]).T / math.sqrt(3)
        normal = N.array([[0, 0, 1]]).T
        correct_reflection = N.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T / math.sqrt(3)
        
        reflection = optics.reflections(dir, normal)
        self.failUnless((reflection == correct_reflection).all(), 
            "Reflection is\n" + str(reflection) + "\nbut should be\n" + str(correct_reflection))
        
class TestTangentRays(unittest.TestCase):
    """Rays tangent to the surface conrinue unchanged"""
    def runTest(self):
        dir = N.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]).T / math.sqrt(2)
        normal = N.array([[0, 0, 1]]).T
        
        reflection = optics.reflections(dir, normal)
        self.failUnless(N.allclose(reflection, dir), 
            "Reflection is\n" + str(reflection) + "\nbut should be\n" + str(dir))

class TestMultipleNormals(unittest.TestCase):
    """When each ray has its own normal, each reflection uses the corresponding 
    normal"""
    def runTest(self):
        dir = N.array([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]).T / math.sqrt(3)
        # The normals are selected to reflect all the rays to the same direction.
        normal = N.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]).T / N.sqrt([1, 2, 3, 2])
        correct_reflection = N.tile([1, 1, 1], (4,1)).T / math.sqrt(3)
        
        reflection = optics.reflections(dir, normal)
        self.failUnless(N.allclose(reflection, correct_reflection), 
            "Reflection is\n" + str(reflection) + "\nbut should be\n" + str(correct_reflection))
        
class TestFresnel(unittest.TestCase):
    """
    Various angles of incidence etc. to test that the Fresnel reflectance is
    correct
    """
    def test_normal_incidence(self):
        """Rays at normal incidence achieve predicted reflectance."""
        dir = N.c_[[0, 0, 1]]
        norm = dir
        R = optics.fresnel(dir, norm, N.r_[1.], N.r_[1.5])
        self.failUnlessAlmostEqual(R, 0.04) # ref [1], page 44
    
    def test_grazing_incidence(self):
        """Grazing rays are not refracted"""
        dir = N.c_[[0, 0, 1]]
        norm = N.c_[[0, 1, 0]]
        R = optics.fresnel(dir, norm, N.r_[1.], N.r_[1.5])
        self.failUnlessAlmostEqual(R, 1.) # ref [1], page 45
    
    def test_no_reflectance(self):
        """With index of refraction = 1, no reflection"""
        dir = N.c_[[0, 1, 1]]/math.sqrt(2)
        norm = N.c_[[0, 1, 0]]
        R = optics.fresnel(dir, norm, N.r_[1.], N.r_[1.])
        self.failUnlessAlmostEqual(R, 0)

if __name__ == "__main__":
    unittest.main()

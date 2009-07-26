# Unit tests for the optics module.

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
        
        reflection = optics.reflections(N.r_[[1]], dir, normal)
        self.failUnless((reflection == correct_reflection).all(), 
            "Reflection is\n" + str(reflection) + "\nbut should be\n" + str(correct_reflection))

class TestMultipleReflections(unittest.TestCase):
    """Four symmetric rays reflect to the correct directions"""
    def runTest(self):
        dir = N.array([[1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]]).T / math.sqrt(3)
        normal = N.array([[0, 0, 1]]).T
        correct_reflection = N.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T / math.sqrt(3)
        
        reflection = optics.reflections(N.r_[[1,1,1,1]], dir, normal)
        self.failUnless((reflection == correct_reflection).all(), 
            "Reflection is\n" + str(reflection) + "\nbut should be\n" + str(correct_reflection))
        
class TestTangentRays(unittest.TestCase):
    """Rays tangent to the surface conrinue unchanged"""
    def runTest(self):
        dir = N.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]).T / math.sqrt(2)
        normal = N.array([[0, 0, 1]]).T
        
        reflection = optics.reflections(N.r_[[1,1,1,1]], dir, normal)
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
        
        reflection = optics.reflections(N.r_[[1,1,1,1]], dir, normal)
        self.failUnless(N.allclose(reflection, correct_reflection), 
            "Reflection is\n" + str(reflection) + "\nbut should be\n" + str(correct_reflection))
        
if __name__ == "__main__":
    unittest.main()

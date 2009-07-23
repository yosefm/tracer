import unittest
import numpy as N
import math

from tracer_engine import TracerEngine
from ray_bundle import RayBundle
from spatial_geometry import general_axis_rotation
from sphere_surface import SphereSurface
from boundary_shape import BoundarySphere
from receiver import Receiver
from object import AssembledObject
from assembly import Assembly
import assembly
import pdb

class TestObjectBuilding1(unittest.TestCase):
    """Tests an object composed of surfaces"""
    def setUp(self):
        self.assembly = Assembly()
        surface1 = SphereSurface(center=N.array([0,0,-1.]), radius=3.)
        surface2 = SphereSurface(center=N.array([0,0,1.]), radius=3.)
        bound = BoundarySphere(center=N.array([0,0,0]), radius=3.)
        
        self.object = AssembledObject()
        self.object.add_surface(surface1)
        self.object.add_surface(surface2)
        self.object.add_boundary(bound)
        self.assembly.add_object(self.object, N.eye(4))

        dir = N.c_[[0,0,1.],[0,0,1.]]
        position = N.c_[[0,0,-3.],[0,0,-1.]]
    
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1,1]])
        self._bund.set_ref_index(N.r_[[1,1]])


    def test_object(self):
        """Tests that the assembly heirarchy works at a basic level"""
        self.engine = TracerEngine(self.assembly, N.r_[[1,1]], N.r_[[1,1]])

        params =  self.engine.ray_tracer(self._bund,1)[0]
        correct_params = N.c_[[0,0,2],[0,0,-2]]

        N.testing.assert_array_almost_equal(params, correct_params)
    
    def test_translation(self):
        """Tests an assembly that has been translated"""
        trans = N.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
        self.assembly.transform_assembly(trans)

        self.engine = TracerEngine(self.assembly, N.r_[[1,1]], N.r_[[1,1]])

        params =  self.engine.ray_tracer(self._bund,1)[0]
        correct_params = N.c_[[0,0,3],[0,0,-1]]

        N.testing.assert_array_almost_equal(params, correct_params)

    def test_rotation_and_translation(self):
        """Tests an assembly that has been translated and rotated"""
        self._bund = RayBundle()
        self._bund.set_vertices(N.c_[[0,-5,1],[0,5,1]])
        self._bund.set_directions(N.c_[[0,1,0],[0,1,0]])
        self._bund.set_energy(N.r_[[1,1]])
        self._bund.set_ref_index(N.r_[[1,1]])

        trans = assembly.generate_transform(N.r_[[1,0,0]], N.pi/2, N.c_[[0,0,1]])
        self.assembly.transform_assembly(trans)

        self.engine = TracerEngine(self.assembly, N.r_[[1,1]], N.r_[[1,1]])

        params =  self.engine.ray_tracer(self._bund,1)[0]
        correct_params = N.c_[[0,-2,1]]

        N.testing.assert_array_almost_equal(params, correct_params)

if __name__ == '__main__':
    unittest.main()


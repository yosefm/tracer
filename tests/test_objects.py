import unittest
import numpy as N
import math

from tracer_engine import TracerEngine
from ray_bundle import RayBundle
from spatial_geometry import general_axis_rotation
from spatial_geometry import generate_transform
from sphere_surface import SphereSurface
from boundary_shape import BoundarySphere
from flat_surface import FlatSurface
from receiver import Receiver
from object import AssembledObject
from assembly import Assembly
import assembly
from paraboloid import Paraboloid
import pdb

class TestObjectBuilding1(unittest.TestCase):
    """Tests an object composed of surfaces"""
    def setUp(self):
        self.assembly = Assembly()
        surface1 = SphereSurface(location=N.array([0,0,-1.]), radius=3.)
        surface2 = SphereSurface(location=N.array([0,0,1.]), radius=3.)
        bound = BoundarySphere(radius=3.)
        
        self.object = AssembledObject()
        self.object.add_surface(surface1)
        self.object.add_surface(surface2)
        self.object.add_boundary(bound)
        self.assembly.add_object(self.object)

        dir = N.c_[[0,0,1.],[0,0,1.]]
        position = N.c_[[0,0,-3.],[0,0,-1.]]
    
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1,1]])
        self._bund.set_ref_index(N.r_[[1,1]])


    def test_object(self):
        """Tests that the assembly heirarchy works at a basic level"""
        self.engine = TracerEngine(self.assembly)

        params =  self.engine.ray_tracer(self._bund,1)[0]
        correct_params = N.c_[[0,0,2],[0,0,-2]]

        N.testing.assert_array_almost_equal(params, correct_params)
    
    def test_translation(self):
        """Tests an assembly that has been translated"""
        trans = N.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
        self.assembly.transform_assembly(trans)

        self.engine = TracerEngine(self.assembly)

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

        trans = generate_transform(N.r_[[1,0,0]], N.pi/2, N.c_[[0,0,1]])
        self.assembly.transform_assembly(trans)

        self.engine = TracerEngine(self.assembly)

        params =  self.engine.ray_tracer(self._bund,1)[0]
        correct_params = N.c_[[0,-2,1]]

        N.testing.assert_array_almost_equal(params, correct_params)

class TestObjectBuilding2(unittest.TestCase):
    """Tests an object composed of two surfaces"""
    def setUp(self):
        self.assembly = Assembly()
        surface1 = FlatSurface(location=N.array([0,0,-1.]), width=5., height=5., mirror=False)
        surface2 = FlatSurface(location=N.array([0,0,1.]), width=5., height=5., mirror=False) 
 
        self.object = AssembledObject()
        self.object.add_surface(surface1)
        self.object.add_surface(surface2)
        self.object.set_ref_index([surface1, surface2], 1.5)
        self.assembly.add_object(self.object)
        
        x = 1/(math.sqrt(2))
        dir = N.c_[[0,-x,x]]
        position = N.c_[[0,1,-2.]]
        
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1.]])
        self._bund.set_ref_index(N.r_[[1.]])

    def test_refraction1(self):
        """Tests the refractive functions after a single intersection"""
        self.engine = TracerEngine(self.assembly)
        ans =  self.engine.ray_tracer(self._bund,1)
        params = N.arctan(ans[1][1]/ans[1][2])
        correct_params = N.r_[-.4908826, 0.785398163]
        N.testing.assert_array_almost_equal(params, correct_params)

    def test_refraction2(self):
        """Tests the refractive functions after two intersections"""
        self.engine = TracerEngine(self.assembly)
        ans = self.engine.ray_tracer(self._bund,2)
        params = N.arctan(ans[1][1]/ans[1][2])
        correct_params = N.r_[-0.7853981]
        N.testing.assert_array_almost_equal(params, correct_params)

class TestAssemblyBuilding3(unittest.TestCase):
    """Tests an assembly composed of objects"""
    def setUp(self):  
        self.assembly = Assembly()

        surface1 = FlatSurface(location=N.array([0,0,-1.]), width=5., height=5., mirror=False)
        surface2 = FlatSurface(location=N.array([0,0,1.]), width=5., height=5., mirror=False)
        self.object1 = AssembledObject() 
        self.object1.add_surface(surface1)
        self.object1.add_surface(surface2)
        self.object1.set_ref_index([surface1, surface2], 1.5)
        
        surface3 = SphereSurface(radius=2.)
        boundary = BoundarySphere(location=N.r_[0,0.,3], radius=3.)
        self.object2 = AssembledObject()
        self.object2.add_surface(surface3)
        self.object2.add_boundary(boundary)
        
        self.transform = generate_transform(N.r_[0,0.,0],0.,N.c_[[0.,0,2]])
        self.assembly.add_object(self.object1)
        self.assembly.add_object(self.object2, self.transform)

        x = 1./(math.sqrt(2))
        dir = N.c_[[0,0,1.],[0,x,x],[0,1.,0]]
        position = N.c_[[0,0,2.],[0,0,2.],[0,0.,2.]]
        
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1.,1.,1.]])
        self._bund.set_ref_index(N.r_[[1.,1.,1.]])
        
    def test_assembly1(self):
        """Tests the assembly after one iteration"""
        self.engine = TracerEngine(self.assembly)
        ans =  self.engine.ray_tracer(self._bund,1)
        params = N.arctan(ans[1][1]/ans[1][2])
        correct_params = N.r_[-0, 0.7853981]

        N.testing.assert_array_almost_equal(params, correct_params)

    def test_assembly2(self):
        """Tests the assembly after two iterations"""
        self.engine = TracerEngine(self.assembly)
        
        params = self.engine.ray_tracer(self._bund,2)[0]
        correct_params = N.c_[[0,0,1],[0,-1,1],[0,-1,1]]
        N.testing.assert_array_almost_equal(params, correct_params)

    def test_assembly3(self):
        """Tests the assembly after three iterations"""
        self.engine = TracerEngine(self.assembly)
        params = self.engine.ray_tracer(self._bund, 3)[0]
        correct_params = N.c_[[0,0,-1],[0,-2.069044,-1],[0,0,-1]]
        N.testing.assert_array_almost_equal(params, correct_params)

class TestAssemblyBuilding4(unittest.TestCase):
    """Tests an assembly composed of objects"""
    def setUp(self):
        self.assembly = Assembly()

        surface1 = Paraboloid()
        boundary = BoundarySphere(radius=3.)
        
        self.object = AssembledObject()
        self.object.add_surface(surface1)
        self.object.add_boundary(boundary)

        self.assembly.add_object(self.object)
       
        x = 1./(math.sqrt(2))
        dir = N.c_[[0,0,-1.]]
        position = N.c_[[0,0,1.]]

        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1.]])
        self._bund.set_ref_index(N.r_[[1.]])

    def test_paraboloid1(self):  
        """Tests a paraboloid"""
        self.engine = TracerEngine(self.assembly)
        params =  self.engine.ray_tracer(self._bund,1)[1]
        correct_params = N.c_[[0,0,1]]
        N.testing.assert_array_almost_equal(params, correct_params)


if __name__ == '__main__':
    unittest.main()


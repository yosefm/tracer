import unittest
import numpy as N
import math

from tracer_engine import TracerEngine
from ray_bundle import RayBundle
from flat_surface import FlatSurface
from spatial_geometry import general_axis_rotation
from sphere_surface import SphereSurface
from boundary_shape import BoundarySphere
from receiver import Receiver
from assembly import Assembly
from object import AssembledObject
import pdb

class TestTraceProtocol1(unittest.TestCase):
    """ 
    Tests intersect_ray and the bundle driver with a single flat surface, not rotated, with 
    a single interation 
    """
    def setUp(self):
        
        dir = N.array([[1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1]]).T/math.sqrt(3)
        position = N.c_[[0,0,1],[1,-1,1],[1,1,1],[-1,1,1]]
       
        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1,1,1,1]])
        self._bund.set_ref_index(N.r_[[1,1,1,1]])
        
        self.assembly = Assembly()
        object = AssembledObject()
        object.add_surface(FlatSurface())
        self.assembly.add_object(object)
        self.engine = TracerEngine(self.assembly, N.r_[[1,1,1,1]],N.r_[[1,1,1,1]])
        
    def test_intersect_ray1(self):
        correct_params = N.r_[[False, True, True, True]]
        params = self.engine.intersect_ray(self._bund)[0]

        N.testing.assert_array_almost_equal(params, correct_params)

    def test_ray_tracer(self):
        params = self.engine.ray_tracer(self._bund,1)[0]
        correct_params = N.c_[[0,0,0],[0,0,0],[0,0,0]]
        
        N.testing.assert_array_almost_equal(params, correct_params)

class TestTraceProtocol2(unittest.TestCase):
    """
    Tests intersect_ray with a flat surface rotated around the x axis 45 degrees
    """
    def setUp(self):
        
        dir = N.c_[[0,0,1],[0,0,-1],[0,-1,-1]]
        position = N.c_[[0,0,1],[0,1,2],[0,0,1]]

        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_ref_index(N.r_[[1,1,1]])

    def test_intersect_ray2(self):
        rot = general_axis_rotation([1,0,0],N.pi/4)
        surface = FlatSurface(rotation=rot,width=4,height=4)
        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surface)
        assembly.add_object(object)
        
        engine = TracerEngine(assembly, N.r_[[1,1,1]], N.r_[[1,1,1]])
        params = engine.intersect_ray(self._bund)[0]
        correct_params = N.r_[[False, True, False]]

        N.testing.assert_array_almost_equal(params, correct_params)

class TestTraceProtocol3(unittest.TestCase):
    """
    Tests intersect_ray and the bundle driver with two rotated planes, with a single iteration
    """
    def setUp(self):
        
        self.x = 1/(math.sqrt(2))
        dir = N.c_[[0,self.x,-self.x],[0,1,0]]
        position = N.c_[[0,0,1],[0,0,1]]

        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_ref_index(N.r_[[1,1,1]])

        rot1 = general_axis_rotation([1,0,0],N.pi/4)
        rot2 = general_axis_rotation([1,0,0],N.pi/(-4))
        surf1 = FlatSurface(rotation=rot1,width=4,height=4)
        surf2 = FlatSurface(rotation=rot2,width=4,height=4)
        energy = N.array([1,1])
        self._bund.set_energy(energy)
        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surf1)
        object.add_surface(surf2)
        assembly.add_object(object)

        self.engine = TracerEngine(assembly, N.r_[[1,1]], N.r_[[1,1,]])
        
    def test_intersect_ray(self):
        params = self.engine.intersect_ray(self._bund)
        correct_params = N.array([[True, True],[False, False]])

        N.testing.assert_array_almost_equal(params,correct_params)    

    def test_ray_tracer1(self):
        params = self.engine.ray_tracer(self._bund, 1)[0]
        correct_params = N.c_[[0,.5,.5],[0,1,1]]

        N.testing.assert_array_almost_equal(params,correct_params)

class TestTraceProtocol4(unittest.TestCase):
    """
    Tests intersect_ray and the bundle driver with two planes, where the rays hit different surfaces
    """
    def setUp(self):

        self.x = 1/(math.sqrt(2))
        dir = N.c_[[0,-self.x,self.x],[0,0,-1]]
        position = N.c_ [[0,2,1],[0,2,1]]

        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_ref_index(N.r_[[1,1,1]]) 

        rot1 = general_axis_rotation([1,0,0],N.pi/4)
        energy = N.array([1,1])
        self._bund.set_energy(energy)

        surf1 = FlatSurface(rotation=rot1,width=10,height=10)
        surf2 = FlatSurface(width=10,height=10)
        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surf1)
        object.add_surface(surf2)
        assembly.add_object(object)
        
        self.engine = TracerEngine(assembly, N.r_[[1,1,1]], N.r_[[1,1,1]])
        
    def test_ray_tracer1(self):
        params = self.engine.ray_tracer(self._bund, 1)[0]
        correct_params = N.c_[[0,1.5,1.5],[0,2,0]]
        
        N.testing.assert_array_almost_equal(params,correct_params)

    def test_ray_tracer2(self):
        params = self.engine.ray_tracer(self._bund, 2)[0]
        correct_params = N.c_[[0,2,2],[0,3,0]]

#        print self.engine.track_ray(self._bund, 1)
        N.testing.assert_array_almost_equal(params,correct_params)

class TestTraceProtocol5(unittest.TestCase):
    """
    Tests a spherical surface
    """
    def setUp(self):
        boundary = BoundarySphere(N.array([0,1.,0]), 1.)
        surface = SphereSurface(center=N.array([0,0,0]))
        self._bund = RayBundle()
        self._bund.set_directions(N.c_[[0,1,0],[0,1,0],[0,-1,0]])
        self._bund.set_vertices(N.c_[[0,-2.,0],[0,0,0],[0,2,0]])
        self._bund.set_energy(N.r_[[1,1,1]])
        self._bund.set_ref_index(N.r_[[1,1,1]])

        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surface)
        object.add_boundary(boundary)
        assembly.add_object(object)
        
        self.engine = TracerEngine(assembly, N.r_[[1,1,1]], N.r_[[1,1,1]])


    def test_ray_tracer1(self):
        params = self.engine.ray_tracer(self._bund, 1)[0]
        correct_params = N.c_[[0,1,0],[0,1,0]]
         
        N.testing.assert_array_almost_equal(params,correct_params)

class TestTraceProtocol6(unittest.TestCase):
    """
    Tests a spherical surface
    """
    def setUp(self):
        boundary1 = BoundarySphere(N.array([0,2.,0]),3.)
        surface1 = SphereSurface(center=N.array([0,0,0]), radius=2)
        boundary2 = BoundarySphere(N.array([0,-5,0]),3)
        surface2 = SphereSurface(center=N.array([0,-2,0]), radius=2)
        self._bund = RayBundle()
        self._bund.set_directions(N.c_[[0,1,0]])
        self._bund.set_vertices(N.c_[[0,-1,0]])
        self._bund.set_energy(N.r_[[1]]) 
        self._bund.set_ref_index(N.r_[[1]])

        assembly = Assembly()
        object1 = AssembledObject()
        object2 = AssembledObject()
        object1.add_surface(surface1)
        object2.add_surface(surface2)
        object1.add_boundary(boundary1)
        object2.add_boundary(boundary2)
        assembly.add_object(object1)
        assembly.add_object(object2)

        self.engine = TracerEngine(assembly, N.r_[[1]], N.r_[[1]])
        
    def test_ray_tracers1(self):
        params = self.engine.ray_tracer(self._bund, 1)[0]
        correct_params = N.c_[[0,2,0]]

        N.testing.assert_array_almost_equal(params,correct_params)

class TestRefraction(unittest.TestCase):
    """Tests refractive properties of a flat surface""" 
    def setUp(self):
        self.x = 1/(math.sqrt(2))
        dir = N.c_[[0,0,-1],[0,-self.x,-self.x]]
        position = N.c_[[0,0,1],[0,1,1]]

        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1,1]])
        self._bund.set_ref_index(N.r_[[1,1]])  
        
        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(FlatSurface())
        assembly.add_object(object)

        self.engine = TracerEngine(assembly, N.r_[[1,1]], N.r_[[1.5, 1.5]])
                       
    def test_intersect_ray1(self):  
       
        correct_params = N.r_[.4908826, 0, -0.785398 ]  
        ans = self.engine.ray_tracer(self._bund, 1)
        params = N.arctan(ans[1][1]/ans[1][2])
       
        N.testing.assert_array_almost_equal(params, correct_params)
  

if __name__ == '__main__':
    unittest.main()

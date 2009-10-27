import unittest
import numpy as N
import math

from tracer_engine import TracerEngine
from ray_bundle import RayBundle
from flat_surface import FlatSurface
from spatial_geometry import general_axis_rotation
from sphere_surface import HemisphereGM
from boundary_shape import BoundarySphere
from receiver import Receiver
from assembly import Assembly
from object import AssembledObject

from surface import Surface
from flat_surface import FlatGeometryManager
import optics_callables as opt

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
        object.add_surface(Surface(FlatGeometryManager(), opt.perfect_mirror))
        self.assembly.add_object(object)
        self.engine = TracerEngine(self.assembly)
        
    def test_intersect_ray1(self):
        params = self.engine.intersect_ray(self._bund)[0]
        self.failUnless(params.all())

    def test_ray_tracer(self):
        """Ray tracer after one iteration returns what the surface would have"""
        params = self.engine.ray_tracer(self._bund,1,.05)[0]
        correct_pts = N.zeros((3,4))
        correct_pts[:2,0] = 1
        
        N.testing.assert_array_almost_equal(params, correct_pts)

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
        surface = Surface(FlatGeometryManager(), opt.perfect_mirror, rotation=rot)
        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surface)
        assembly.add_object(object)
        
        engine = TracerEngine(assembly)
        params = engine.intersect_ray(self._bund)[0]
        correct_params = N.r_[[False, True, True]]

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
        energy = N.array([1,1])
        self._bund.set_energy(energy)

        rot1 = general_axis_rotation([1,0,0],N.pi/4)
        rot2 = general_axis_rotation([1,0,0],N.pi/(-4))
        surf1 = Surface(FlatGeometryManager(), opt.perfect_mirror, rotation=rot1)
        surf2 = Surface(FlatGeometryManager(), opt.perfect_mirror, rotation=rot2)
        
        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surf1)
        object.add_surface(surf2)
        assembly.add_object(object)

        self.engine = TracerEngine(assembly)
        
    def test_intersect_ray(self):
        params = self.engine.intersect_ray(self._bund)
        correct_params = N.array([[True, True],[False, False]])

        N.testing.assert_array_almost_equal(params,correct_params)    

    def test_ray_tracer1(self):
        params = self.engine.ray_tracer(self._bund, 1,.05)[0]
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

        surf1 = Surface(FlatGeometryManager(), opt.perfect_mirror, rotation=rot1)
        surf2 = Surface(FlatGeometryManager(), opt.perfect_mirror)
        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surf1)
        object.add_surface(surf2)
        assembly.add_object(object)
        
        self.engine = TracerEngine(assembly)
        
    def test_ray_tracer1(self):
        params = self.engine.ray_tracer(self._bund, 1,.05)[0]
        correct_params = N.c_[[0,1.5,1.5],[0,2,0]]
        
        N.testing.assert_array_almost_equal(params,correct_params)

    def test_ray_tracer2(self):
        params = self.engine.ray_tracer(self._bund, 2,.05)[0]
        correct_params = N.c_[[0,2,2],[0,3,0]]
        
        N.testing.assert_array_almost_equal(params,correct_params)
    
    def test_too_much_intersections(self):
        """The tracer stops when all rays are escaping"""
        params = self.engine.ray_tracer(self._bund, 42, 0.05)[0]
        correct_params = N.array([]).reshape(3,0)

        N.testing.assert_array_almost_equal(params,correct_params)

class TestTraceProtocol5(unittest.TestCase):
    """
    Tests a spherical surface
    """
    def setUp(self):
        surface = Surface(HemisphereGM(1.), opt.perfect_mirror,
            rotation=general_axis_rotation(N.r_[1,0,0], N.pi))
        self._bund = RayBundle()
        self._bund.set_directions(N.c_[[0,1,0],[0,1,0],[0,-1,0]])
        self._bund.set_vertices(N.c_[[0,-2.,0],[0,0,0],[0,2,0]])
        self._bund.set_energy(N.r_[[1,1,1]])
        self._bund.set_ref_index(N.r_[[1,1,1]])

        assembly = Assembly()
        object = AssembledObject()
        object.add_surface(surface)
        assembly.add_object(object)
        
        self.engine = TracerEngine(assembly)

    def test_ray_tracer1(self):
        params = self.engine.ray_tracer(self._bund, 1, .05)[0]
        correct_params = N.c_[[0,1,0],[0,1,0],[0,1,0]]
         
        N.testing.assert_array_almost_equal(params,correct_params)

class TestTraceProtocol6(unittest.TestCase):
    """
    Tests a spherical surface
    """
    def setUp(self):
        surface1 = Surface(HemisphereGM(2.), opt.perfect_mirror,
            rotation=general_axis_rotation(N.r_[1,0,0], N.pi/2.))
        surface2 = Surface(HemisphereGM(2.), opt.perfect_mirror, 
            location=N.array([0,-2,0]), 
            rotation=general_axis_rotation(N.r_[1,0,0], -N.pi/2.))
        
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
        assembly.add_object(object1)
        assembly.add_object(object2)

        self.engine = TracerEngine(assembly)
        
    def test_ray_tracers1(self):
        params = self.engine.ray_tracer(self._bund, 1, .05)[0]
        correct_params = N.c_[[0,2,0]]

        N.testing.assert_array_almost_equal(params,correct_params)

if __name__ == '__main__':
    unittest.main()

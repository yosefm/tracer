# -*- coding: utf-8 -*-
import unittest
import numpy as N
import math

from tracer.tracer_engine import TracerEngine
from tracer.ray_bundle import RayBundle
from tracer.spatial_geometry import general_axis_rotation, generate_transform
from tracer.sphere_surface import HemisphereGM, CutSphereGM
from tracer.boundary_shape import BoundarySphere
from tracer.object import AssembledObject
from tracer.assembly import Assembly
from tracer import assembly
from tracer.paraboloid import Paraboloid

from tracer.surface import Surface
from tracer import flat_surface
from tracer import optics_callables

class TestObjectBuilding1(unittest.TestCase):
    """Tests an object composed of sphere surfaces"""
    def setUp(self):
        self.assembly = Assembly()
        surface1 = Surface(HemisphereGM(3.), optics_callables.perfect_mirror, 
            location=N.array([0,0,-1.]),
            rotation=general_axis_rotation(N.r_[1,0,0], N.pi))
        surface2 = Surface(HemisphereGM(3.), optics_callables.perfect_mirror, 
            location=N.array([0,0,1.]))
        
        self.object = AssembledObject()
        self.object.add_surface(surface1)
        self.object.add_surface(surface2)
        self.assembly.add_object(self.object)

        dir = N.c_[[0,0,1.],[0,0,1.]]
        position = N.c_[[0,0,-3.],[0,0,-1.]]
    
        self._bund = RayBundle(position, dir, energy=N.ones(2))
    
    def test_object(self):
        """Tests that the assembly heirarchy works at a basic level"""
        self.engine = TracerEngine(self.assembly)

        inters = self.engine.ray_tracer(self._bund,1,.05)[0]
        correct_inters = N.c_[[0,0,2],[0,0,-2]]

        N.testing.assert_array_almost_equal(inters, correct_inters)
    
    def test_translation(self):
        """Tests an assembly that has been translated"""
        trans = N.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])
        self.assembly.transform_children(trans)

        self.engine = TracerEngine(self.assembly)

        params =  self.engine.ray_tracer(self._bund,1,.05)[0]
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
        self.assembly.transform_children(trans)

        self.engine = TracerEngine(self.assembly)

        params =  self.engine.ray_tracer(self._bund,1,.05)[0]
        correct_params = N.c_[[0,-2,1]]

        N.testing.assert_array_almost_equal(params, correct_params)

class TestObjectBuilding2(unittest.TestCase):
    """Tests an object composed of two flat surfaces"""
    flat_only = True
    def setUp(self):
        self.assembly = Assembly()
        surface1 = Surface(flat_surface.FlatGeometryManager(),
            optics_callables.RefractiveHomogenous(1., 1.5),
            location=N.array([0,0,-1.]))
        surface2 = Surface(flat_surface.FlatGeometryManager(),
            optics_callables.RefractiveHomogenous(1.5, 1.),
            location=N.array([0,0,1.]))
        
        self.object = AssembledObject(surfs=[surface1, surface2])
        self.assembly.add_object(self.object)
        
        x = 1/(math.sqrt(2))
        dir = N.c_[[0,-x,x]]
        position = N.c_[[0,1,-2.]]
        self._bund = RayBundle(position, dir, energy=N.r_[1.], ref_index=N.r_[1.])

    def test_refraction1(self):
        """Tests the refractive functions after a single intersection"""
        self.engine = TracerEngine(self.assembly)
        ans =  self.engine.ray_tracer(self._bund,1,.05)
        params = N.arctan(ans[1][1]/ans[1][2])
        correct_params = N.r_[0.785398163, -.4908826]
        N.testing.assert_array_almost_equal(params, correct_params)

    def test_refraction2(self):
        """Tests the refractive functions after two intersections"""
        self.engine = TracerEngine(self.assembly)
        ans = self.engine.ray_tracer(self._bund,2,.05)
        params = N.arctan(ans[1][1]/ans[1][2])
        correct_params = N.r_[-0.7853981]
        N.testing.assert_array_almost_equal(params, correct_params)

class TestAssemblyBuilding3(unittest.TestCase):
    """Tests an assembly composed of objects that are transformed rel. the assembly"""
    def setUp(self):  
        self.assembly = Assembly()

        surface1 = Surface(flat_surface.FlatGeometryManager(), 
            optics_callables.RefractiveHomogenous(1., 1.5),
            location=N.array([0,0,-1.]))
        surface2 = Surface(flat_surface.FlatGeometryManager(), 
            optics_callables.RefractiveHomogenous(1., 1.5),
            location=N.array([0,0,1.]))
        
        self.object1 = AssembledObject() 
        self.object1.add_surface(surface1)
        self.object1.add_surface(surface2)
        
        boundary = BoundarySphere(location=N.r_[0,0.,3], radius=3.)
        surface3 = Surface(CutSphereGM(2., boundary), optics_callables.perfect_mirror)
        self.object2 = AssembledObject()
        self.object2.add_surface(surface3)
    
        self.transform = generate_transform(N.r_[1,0.,0],0.,N.c_[[0.,0,2]])
        self.assembly.add_object(self.object1)
        self.assembly.add_object(self.object2, self.transform)
        
        x = 1./(math.sqrt(2))
        dir = N.c_[[0,1.,0.],[0,x,x],[0,0,1.]]
        position = N.c_[[0,0,2.],[0,0,2.],[0,0.,2.]]
        self._bund = RayBundle(position, dir, energy=N.ones(3), ref_index=N.ones(3))
        
    def test_assembly1(self):
        """Tests the assembly after one iteration"""
        self.engine = TracerEngine(self.assembly)
        ans =  self.engine.ray_tracer(self._bund,1,.05)
        params = N.arctan(ans[1][1]/ans[1][2])
        correct_params = N.r_[0.7853981, 0]

        N.testing.assert_array_almost_equal(params, correct_params)

    def test_assembly2(self):
        """Tests the assembly after two iterations"""
        self.engine = TracerEngine(self.assembly)
        params = self.engine.ray_tracer(self._bund,2,.05)[0]
        correct_params = N.c_[[0,-1,1], [0,-1,1],[0,0,1]]
        N.testing.assert_array_almost_equal(params, correct_params)

    def test_assembly3(self):      
        """Tests the assembly after three iterations"""  
        self.engine = TracerEngine(self.assembly)
        params = self.engine.ray_tracer(self._bund, 3,.05)[0]
        correct_params = N.c_[[0,-2.069044,-1],[0,0,-1]]

        N.testing.assert_array_almost_equal(params, correct_params)

class TestAssemblyBuilding4(unittest.TestCase):
    """Tests an assembly composed of objects"""
    def setUp(self):
        self.assembly = Assembly()
        surface1 = Surface(Paraboloid(), optics_callables.perfect_mirror)
        self.object = AssembledObject()
        self.object.add_surface(surface1)
        self.assembly.add_object(self.object)
       
        x = 1./(math.sqrt(2))  
        dir = N.c_[[0,0,-1.],[0,x,-x]]
        position = N.c_[[0,0,1.],[0,0,1.]]
        self._bund = RayBundle(position, dir, energy=N.ones(2), ref_index=N.ones(2))

    def test_paraboloid1(self):  
        """Tests a paraboloid"""
        
        self.engine = TracerEngine(self.assembly)
        params =  self.engine.ray_tracer(self._bund,1,.05)[0]
        correct_params = N.c_[[0,0,0],[0,0.618033989, 0.381966011]]
        N.testing.assert_array_almost_equal(params, correct_params)

class TestNestedAssemblies(unittest.TestCase):
    """
    Create an assembly within an assembly, with an object, and check that 
    all transformation activities are handled correctly.
    """
    flat_only = True
    def setUp(self):
        self.eighth_circle_trans = generate_transform(N.r_[1., 0, 0], N.pi/4, 
            N.c_[[0., 1, 0]])
        
        self.surf = Surface(flat_surface.FlatGeometryManager(), \
            optics_callables.perfect_mirror)
        self.obj = AssembledObject(surfs=[self.surf])
        self.sub_assembly = Assembly()
        self.sub_assembly.add_object(self.obj, self.eighth_circle_trans)
        self.assembly = Assembly()
        self.assembly.add_assembly(self.sub_assembly, self.eighth_circle_trans)
    
    def test_initial_transforms(self):
        """Initial consrtruction yielded correct permanent and temporary transforms"""
        quarter_circle_trans = N.dot(self.eighth_circle_trans, self.eighth_circle_trans)
        
        # Surface transforms:
        N.testing.assert_array_almost_equal(self.surf._transform, N.eye(4))
        N.testing.assert_array_almost_equal(self.surf._temp_frame, quarter_circle_trans)
        
        # Object transform:
        N.testing.assert_array_almost_equal(self.obj.get_transform(),
            self.eighth_circle_trans)
        
        # Subassembly transform:
        N.testing.assert_array_almost_equal(self.sub_assembly.get_transform(),
            self.eighth_circle_trans)
    
    def test_retransform_object(self):
        """Changing an object's transform yield's correct resaults after retransform"""
        self.obj.set_transform(N.eye(4))
        self.assembly.transform_children()
        
        # Surface transforms:
        N.testing.assert_array_almost_equal(self.surf._transform, N.eye(4))
        N.testing.assert_array_almost_equal(self.surf._temp_frame, 
            self.eighth_circle_trans)
        
        # Object transform:
        N.testing.assert_array_almost_equal(self.obj.get_transform(),
            N.eye(4))
        
        # Subassembly transform:
        N.testing.assert_array_almost_equal(self.sub_assembly.get_transform(),
            self.eighth_circle_trans)
    
    def test_retransform_subassembly(self):
        """Changing an assembly's transform yield's correct resaults after retransform"""
        self.sub_assembly.set_transform(N.eye(4))
        self.assembly.transform_children()

        # Surface transforms:
        N.testing.assert_array_almost_equal(self.surf._transform, N.eye(4))
        N.testing.assert_array_almost_equal(self.surf._temp_frame,
            self.eighth_circle_trans)
        
        # Object transform:
        N.testing.assert_array_almost_equal(self.obj.get_transform(),
            self.eighth_circle_trans)
        
        # Subassembly transform:
        N.testing.assert_array_almost_equal(self.sub_assembly.get_transform(),
            N.eye(4))
    
    def test_interface(self):
        """Can call getters on an assembly etc."""
        subs = self.assembly.get_assemblies()
        self.assertEqual(len(subs), 1)
        self.assertTrue(subs[0] is self.sub_assembly)

if __name__ == '__main__':
    unittest.main()


import unittest
import numpy as N
import math

from tracer.tracer_engine import TracerEngine
from tracer.ray_bundle import RayBundle
from tracer.spatial_geometry import translate, generate_transform
from tracer.sphere_surface import CutSphereGM
from tracer.boundary_shape import BoundarySphere
from tracer.flat_surface import FlatGeometryManager
from tracer.object import AssembledObject
from tracer.assembly import Assembly

import tracer.optics_callables as opt
from tracer.surface import Surface

from tracer.models import homogenizer

class TestTree(unittest.TestCase):
    """Tests an assembly composed of objects"""
    def setUp(self):
        self.assembly = Assembly()

        surface1 = Surface(FlatGeometryManager(), opt.perfect_mirror)
        self.object1 = AssembledObject()
        self.object1.add_surface(surface1)  
        
        boundary = BoundarySphere(location=N.r_[0,0.,3], radius=3.)
        surface3 = Surface(CutSphereGM(2., boundary), opt.perfect_mirror)
        self.object2 = AssembledObject()
        self.object2.add_surface(surface3)

        self.transform1 = generate_transform(N.r_[1.,0,0], N.pi/4, N.c_[[0,0,-1.]])
        self.transform2 = translate(0., 0., 2.)
        self.assembly.add_object(self.object1, self.transform1)
        self.assembly.add_object(self.object2, self.transform2)
      
    def test_tree1(self):
        """Tests that the tracing tree works, with three rays"""
        x = 1./(math.sqrt(2))
        dir = N.c_[[0,x,x],[0,-x,x],[0,0,1.]]
        position = N.c_[[0,0,2.],[0,0,2.],[0,0.,2.]]

        bund = RayBundle()
        bund.set_vertices(position)
        bund.set_directions(dir)
        bund.set_energy(N.r_[[1.,1.,1.]])
        bund.set_ref_index(N.r_[[1.,1.,1.]])

        self.engine = TracerEngine(self.assembly)

        self.engine.ray_tracer(bund,3,.05)[0]
        params = self.engine.get_parents_from_tree()
        correct_params = [N.r_[0,1,2],N.r_[1,2],N.r_[0]]
        N.testing.assert_equal(params, correct_params)

    def test_tree2(self):
        """Tests that the tracing tree works, with a new set of rays"""
        x = 1./(math.sqrt(2))
        position = N.c_[[0,0.,-5.],[0,0.,2.],[0,2.,-5.],[0,0.,0],[0,0,2.]]
        dir = N.c_[[0,0,1.],[0,x,-x],[0,0,-1.],[0,0,1.],[0,-x,x]]
        
        bund = RayBundle()
        bund.set_vertices(position)
        bund.set_directions(dir)
        bund.set_energy(N.r_[[1.,1.,1.,1,1.]])
        bund.set_ref_index(N.r_[[1.,1.,1.,1,1.]])

        self.engine = TracerEngine(self.assembly)

        self.engine.ray_tracer(bund,3,.05)[0]
        params = self.engine.get_parents_from_tree()
        correct_params = [N.r_[0,1,3,4],N.r_[2,3,1],N.r_[2,1]]
        N.testing.assert_equal(params, correct_params)

class TestTree2(unittest.TestCase):
    """Tests the tracing tree with a refractive surface"""  
    def setUp(self):
        self.assembly = Assembly()

        surface1 = Surface(FlatGeometryManager(), 
            opt.RefractiveHomogenous(1., 1.5),
            location=N.array([0,0,-1.]))
        surface2 = Surface(FlatGeometryManager(), 
            opt.RefractiveHomogenous(1., 1.5),
            location=N.array([0,0,1.]))
        object1 = AssembledObject(surfs=[surface1, surface2])

        boundary = BoundarySphere(location=N.r_[0,0.,3], radius=3.)
        surface3 = Surface(CutSphereGM(2., boundary), opt.perfect_mirror)
        object2 = AssembledObject(surfs=[surface3], 
            transform=translate(0., 0., 2.))
        
        self.assembly = Assembly(objects=[object1, object2])

        x = 1./(math.sqrt(2))
        dir = N.c_[[0,1.,0.],[0,x,x],[0,0,1.]]
        position = N.c_[[0,0,2.],[0,0,2.],[0,0.,2.]]

        self._bund = RayBundle()
        self._bund.set_vertices(position)
        self._bund.set_directions(dir)
        self._bund.set_energy(N.r_[[1.,1.,1.]])
        self._bund.set_ref_index(N.r_[[1.,1.,1.]])

        self.engine = TracerEngine(self.assembly)

    def test_assembly3(self):
        """Tests the assembly after three iterations"""
        self.engine.ray_tracer(self._bund,3,.05)[0]
        params = self.engine.get_parents_from_tree()
        correct_params = [N.r_[1,2],N.r_[0,0,1],N.r_[1,2]]
        N.testing.assert_equal(params, correct_params)
    
    def test_no_tree(self):
        """Running with tree=False only saves last bundle."""
        self.engine.ray_tracer(self._bund, 3, .05, tree=False)
        parents = self.engine.get_parents_from_tree()
        N.testing.assert_equal(parents, [N.r_[1,2]])

class TestRayCulling(unittest.TestCase):
    def setUp(self):
        asm = homogenizer.rect_homogenizer(1., 1., 1.2, 1.)
        
        self.bund = RayBundle()
        # 4 rays starting somewhat above (+z) the homogenizer
        pos = N.zeros((3,4))
        pos[2] = 1.6
        self.bund.set_vertices(pos)

        # One ray going to each wall, bearing down (-z):
        dir = N.c_[[1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1]]/N.sqrt(2)
        self.bund.set_directions(dir)

        # Laborious setup details:
        self.bund.set_energy(N.ones(4)*4.)
        self.bund.set_ref_index(N.ones(4))
        
        self.engine = TracerEngine(asm)
    
    def test_final_order(self):
        """Rays come out of a homogenizer with the right parents order"""
        self.engine.ray_tracer(self.bund, 300, .05)
        parents = self.engine.get_parents_from_tree()
        correct_parents = [N.r_[0, 1, 2, 3], N.r_[1, 0, 3, 2]]
        N.testing.assert_equal(parents, correct_parents)

if __name__ == '__main__':
    unittest.main()

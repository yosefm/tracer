import unittest
import numpy as N
import math

from tracer.tracer_engine import TracerEngine
from tracer.ray_bundle import RayBundle
from tracer.spatial_geometry import translate, generate_transform
from tracer.sphere_surface import CutSphereGM
from tracer.boundary_shape import BoundarySphere
from tracer.flat_surface import FlatGeometryManager, RectPlateGM
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
        bund = RayBundle(position, dir, energy=N.ones(3))

        self.engine = TracerEngine(self.assembly)

        self.engine.ray_tracer(bund,3,.05)[0]
        params = self.engine.tree.ordered_parents()
        correct_params = [N.r_[0,1,2],N.r_[1,2],N.r_[0]]
        N.testing.assert_equal(params, correct_params)

    def test_tree2(self):
        """Tests that the tracing tree works, with a new set of rays"""
        x = 1./(math.sqrt(2))
        position = N.c_[[0,0.,-5.],[0,0.,2.],[0,2.,-5.],[0,0.,0],[0,0,2.]]
        dir = N.c_[[0,0,1.],[0,x,-x],[0,0,-1.],[0,0,1.],[0,-x,x]]
        bund = RayBundle(position, dir, energy=N.ones(5))

        self.engine = TracerEngine(self.assembly)
        self.engine.ray_tracer(bund,3,.05)[0]
        
        params = self.engine.tree.ordered_parents()
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
        self._bund = RayBundle(position, dir, ref_index=N.ones(3), energy=N.ones(3))

        self.engine = TracerEngine(self.assembly)

    def test_assembly3(self):
        """Tests the assembly after three iterations"""
        self.engine.ray_tracer(self._bund,3,.05)[0]
        params = self.engine.tree.ordered_parents()
        correct_params = [N.r_[1,2],N.r_[0,0,1,1],N.r_[1,2,1,2]]
        N.testing.assert_equal(params, correct_params)
    
    def test_no_tree(self):
        """Running with tree=False only saves last bundle."""
        self.engine.ray_tracer(self._bund, 3, .05, tree=False)
        parents = self.engine.tree.ordered_parents()
        self.failUnlessEqual(len(parents), 0)

class TestRayCulling(unittest.TestCase):
    def setUp(self):
        asm = homogenizer.rect_homogenizer(1., 1., 1.2, 1.)
        
        # 4 rays starting somewhat above (+z) the homogenizer
        pos = N.zeros((3,4))
        pos[2] = 1.6

        # One ray going to each wall, bearing down (-z):
        dir = N.c_[[1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1]]/N.sqrt(2)

        self.bund = RayBundle(pos, dir, ref_index=N.ones(4), energy=N.ones(4)*4.)
        self.engine = TracerEngine(asm)
    
    def test_final_order(self):
        """Rays come out of a homogenizer with the right parents order"""
        self.engine.ray_tracer(self.bund, 300, .05)
        parents = self.engine.tree.ordered_parents()
        correct_parents = [N.r_[0, 1, 2, 3], N.r_[1, 0, 3, 2]]
        N.testing.assert_equal(parents, correct_parents)

class TestLowEnergyParenting(unittest.TestCase):
    """
    A case where rays are absorbed by a first surface, and moved to back of 
    bundle.
    """
    def setUp(self):
        absorptive = Surface(RectPlateGM(1., 1.), opt.Reflective(1.),
            location=N.r_[ 0.5, 0., 1.])
        reflective = Surface(RectPlateGM(1., 1.), opt.Reflective(0.),
            location=N.r_[-0.5, 0., 1.])
        self.assembly = Assembly(
            objects=[AssembledObject(surfs=[absorptive, reflective])])
        
        # 4 rays: two toward absorptive, two toward reflective.
        pos = N.zeros((3,4))
        pos[0] = N.r_[0.5, 0.25, -0.25, -0.5]
        direct = N.zeros((3,4))
        direct[2] = 1.
        self.bund = RayBundle(pos, direct, energy=N.ones(4))
    
    def test_absorbed_to_back(self):
        """Absorbed rays moved to back of recorded bundle"""
        engine = TracerEngine(self.assembly)
        engine.ray_tracer(self.bund, 300, .05)
        
        parents = engine.tree.ordered_parents()
        N.testing.assert_equal(parents, [N.r_[2,3, 0, 1]])

if __name__ == '__main__':
    unittest.main()


import unittest
import numpy as np

from tracer.models.triangulated_surface import TriangulatedSurface
from tracer.models.one_sided_mirror import rect_one_sided_mirror

from tracer.assembly import Assembly
from tracer.optics_callables import perfect_mirror
from tracer.ray_bundle import RayBundle
from tracer.tracer_engine import TracerEngine

class TestFaceSet(unittest.TestCase):
    def test_pyramid(self):
        """A simple right-pyramid triangular mesh"""
        # Face set:
        verts = np.vstack((np.zeros(3), np.eye(3))) # origin + unit along each axis
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        assembly = Assembly(
            objects=[TriangulatedSurface(verts, faces, perfect_mirror)])
        
        # Ray bundle:
        pos = np.c_[[1.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5], [0.5, 0.5, 1.5]]
        direct = np.c_[[-1., 0., 0.], [1., 0., 0.],
            [0., -1., 0.], [0., 1., 0.],
            [0., 0., 1.], [0., 0., -1.]]
        rayb = RayBundle(pos, direct, energy=np.ones(6))

        engine = TracerEngine(assembly)
        verts = engine.ray_tracer(rayb, 1, .05)[0]
        
        p = engine.tree[-1].get_parents()
        zrays = (p >= 4)
        np.testing.assert_array_equal(verts[:,zrays],
            np.tile(np.c_[[0.5, 0.5, 0.]], (1,4)) )
        yrays = (p == 2) | (p ==3) # Only 2 rays here. Edge degeneracy? maybe.
        np.testing.assert_array_equal(verts[:,yrays],
            np.tile(np.c_[[0.5, 0., 0.5]], (1,4)) )
        xrays = (p < 2)
        np.testing.assert_array_equal(verts[:,xrays],
            np.tile(np.c_[[0., 0.5, 0.5]], (1,4)) )
    
    def test_tetrahedron(self):
        """Triangular mesh with oblique triangles"""
        # Face set:
        theta = np.arange(np.pi/2., np.pi*2, 2*np.pi/3)
        base_verts = np.vstack(( np.cos(theta), np.sin(theta), np.ones(3) )).T
        verts = np.vstack((np.zeros(3), base_verts))
        
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        fset = TriangulatedSurface(verts, faces, perfect_mirror)
        
        # Flat floor:
        floor = rect_one_sided_mirror(5., 5., 1.)
        floor.set_location(np.r_[0., 0., 1.])
        assembly = Assembly(objects=[fset, floor])
        
        # Ray bundle of 3 rays starting at equal angles around the tetrahedron:
        theta -= np.pi/3.
        pos = np.vstack((np.cos(theta), np.sin(theta), np.ones(3)*0.2)) * 0.2
        direct = np.vstack(( np.zeros((2,3)), np.ones(3) ))
        rayb = RayBundle(pos, direct, energy=np.ones(6))
        
        # Check that the points on the floor describe an isosceles.
        engine = TracerEngine(assembly)
        engine.ray_tracer(rayb, 2, .05)[0]
        verts = engine.tree[-1].get_vertices()
        sizes = np.sqrt(
            np.sum((verts - np.roll(verts, 1, axis=1))**2, axis=0))
        
        self.assertAlmostEqual(sizes[0], sizes[1])
        self.assertAlmostEqual(sizes[2], sizes[1])

    def test_move_vertices(self):
        """Moving a vertex on a face set replaces the touching surfaces."""
        # Let's create a hexahedron, then move one vertex to make a
        # tetrahedron.
        
        # Face set:
        verts = np.vstack((np.zeros(3), np.eye(3), np.ones(3))) 
        # origin + unit along each axis
        faces = np.array([
            [0, 1, 2], [0, 1, 3], [0, 2, 3], # bottom tetrahedron
            [4, 1, 2], [4, 1, 3], [4, 2, 3]])
        trisurf = TriangulatedSurface(verts, faces, perfect_mirror)
        assembly = Assembly(objects=[trisurf])
        
        # Transformation:
        trisurf.move_vertices(np.r_[4], np.ones((1,3))*np.sqrt(2))
        
        # Ray bundle:
        pos = np.c_[[1.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
            [0.5, 1.5, 0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5], [0.5, 0.5, 1.5]]
        direct = np.c_[[-1., 0., 0.], [1., 0., 0.],
            [0., -1., 0.], [0., 1., 0.],
            [0., 0., 1.], [0., 0., -1.]]
        rayb = RayBundle(pos, direct, energy=np.ones(6))
        
        engine = TracerEngine(assembly)
        verts = engine.ray_tracer(rayb, 1, .05)[0]
        
        p = engine.tree[-1].get_parents()
        zrays = (p >= 4)
        np.testing.assert_array_equal(verts[:,zrays],
            np.tile(np.c_[[0.5, 0.5, 0.]], (1,4)) )
        yrays = (p == 2) | (p ==3) # Only 2 rays here. Edge degeneracy? maybe.
        np.testing.assert_array_equal(verts[:,yrays],
            np.tile(np.c_[[0.5, 0., 0.5]], (1,4)) )
        xrays = (p < 2)
        np.testing.assert_array_equal(verts[:,xrays],
            np.tile(np.c_[[0., 0.5, 0.5]], (1,4)) )

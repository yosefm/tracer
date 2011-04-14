
import unittest
import numpy as np

from tracer.models.triangulated_surface import TriangulatedSurface
from tracer.assembly import Assembly
from tracer.optics_callables import perfect_mirror
from tracer.ray_bundle import RayBundle
from tracer.tracer_engine import TracerEngine

class TestFaceSet(unittest.TestCase):
    def test_pyramid(self):
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
        yrays = (p == 2) | (p ==3)
        np.testing.assert_array_equal(verts[:,yrays],
            np.tile(np.c_[[0.5, 0., 0.5]], (1,4)) )
        xrays = (p < 2) # Only 3 rays here. Edge degeneracy? maybe.
        np.testing.assert_array_equal(verts[:,xrays],
            np.tile(np.c_[[0., 0.5, 0.5]], (1,3)) )

# Test configurations of the spherical lens object.

import unittest
import numpy as N

from tracer.ray_bundle import RayBundle
from tracer.assembly import Assembly
from tracer.models.spherical_lens import SphericalLens
from tracer.models.one_sided_mirror import rect_one_sided_mirror
from tracer.tracer_engine import TracerEngine
from tracer.spatial_geometry import translate

class Biconvex(unittest.TestCase):
    def setUp(self):
        self.lens = SphericalLens(diameter=1., depth=0.1, R1=10., R2=-10., 
            refr_idx=1.5)
    
    def test_focal_length(self):
        """Biconvex calculated focal length is as expected from the lensmaker equation"""
        self.failUnlessEqual(self.lens.focal_length(), 2./(0.2 - 0.05/150))
    
    def test_paraxial_ray(self):
        """A paraxial ray reaches the focus"""
        rb = RayBundle(N.c_[[0., 0.001, 1.]], N.c_[[0., 0., -1.]], 
            energy=N.r_[1.], ref_index=N.r_[1.])
        screen = rect_one_sided_mirror(5, 5)
        f = self.lens.focal_length()
        screen.set_transform(translate(0, 0, -f))
        
        e = TracerEngine(Assembly([self.lens, screen]))
        vert, _ = e.ray_tracer(rb, 3, 1e-6)
        
        self.failUnlessAlmostEqual(vert[1,2], 0, 4)
    
    def test_cylinder(self):
        """The bounding cylinder exists for biconvex lens"""
        f = self.lens.focal_length()
        rb = RayBundle(N.c_[[0., 0., 0.08]], N.c_[[1., 0., 0.]],
            energy=N.r_[1.], ref_index=N.r_[1.5])
        
        e = TracerEngine(Assembly([self.lens]))
        verts, dirs = e.ray_tracer(rb, 1, 1e-6)

        N.testing.assert_array_equal(verts, N.tile(N.c_[[0.5, 0., 0.08]], (1,2)))
        N.testing.assert_array_equal(dirs, N.c_[[-1., 0., 0.], [1., 0., 0.]])

class Biconcave(unittest.TestCase):
    def setUp(self):
        self.lens = SphericalLens(diameter=1., depth=0.1, R1=-10., R2=10.,
            refr_idx=1.5)
    
    def test_focal_length(self):
        """Biconcave calculated focal length is as expected from the lensmaker equation"""
        self.failUnlessEqual(self.lens.focal_length(), 2./(-0.2 - 0.05/150))
    
    def test_image_size(self):
        """Image size of an object imaged by a biconcave lens matches theory"""
        origin = N.c_[[0., 0.001, 1.]]
        direct = -origin/N.linalg.norm(origin)
        rb = RayBundle(origin, direct, energy=N.r_[1.], ref_index=N.r_[1.])
        
        # Magnification, see [1] p. 26.
        f = self.lens.focal_length()
        m = f/(origin[2] + f)
        
        # Image location, [ibid]:
        loc = origin[2]*m
        screen = rect_one_sided_mirror(5, 5)
        screen.set_transform(translate(0, 0, -loc))
        
        e = TracerEngine(Assembly([self.lens, screen]))
        vert, _ = e.ray_tracer(rb, 3, 1e-6)
        
        self.failUnlessAlmostEqual(vert[1,2], -m*origin[1], 4)
    
    def test_cylinder(self):
        """The bounding cylinder exists for biconcave lens"""
        f = self.lens.focal_length()
        rb = RayBundle(N.c_[[0., 0., 0.08]], N.c_[[1., 0., 0.]],
            energy=N.r_[1.], ref_index=N.r_[1.5])
        
        e = TracerEngine(Assembly([self.lens]))
        verts, dirs = e.ray_tracer(rb, 1, 1e-6)

        N.testing.assert_array_equal(verts, N.tile(N.c_[[0.5, 0., 0.08]], (1,2)))
        N.testing.assert_array_equal(dirs, N.c_[[-1., 0., 0.], [1., 0., 0.]])

class PlanoConvex(unittest.TestCase):
    def setUp(self):
        self.lens = SphericalLens(diameter=1., depth=0.05, R1=10., R2=N.inf,
            refr_idx=1.5)
    
    def test_focal_length(self):
        """PlanoConvex calculated focal length is as expected from the lensmaker equation"""
        self.failUnlessEqual(self.lens.focal_length(), 20.)
    
    def test_paraxial_ray(self):
        """A paraxial ray reaches the focus of a planoconvex lens"""
        rb = RayBundle(N.c_[[0., 0.001, 1.]], N.c_[[0., 0., -1.]], 
            energy=N.r_[1.], ref_index=N.r_[1.])
        screen = rect_one_sided_mirror(5, 5)
        f = self.lens.focal_length()
        screen.set_transform(translate(0, 0, -f))
        
        e = TracerEngine(Assembly([self.lens, screen]))
        vert, _ = e.ray_tracer(rb, 3, 1e-6)
        
        self.failUnlessAlmostEqual(vert[1,2], 0, 4)
    
    def test_cylinder(self):
        """The bounding cylinder exists for planoconvex lens"""
        f = self.lens.focal_length()
        rb = RayBundle(N.c_[[0., 0., 0.001]], N.c_[[1., 0., 0.]],
            energy=N.r_[1.], ref_index=N.r_[1.5])

        e = TracerEngine(Assembly([self.lens]))
        verts, dirs = e.ray_tracer(rb, 1, 1e-6)

        N.testing.assert_array_equal(verts, N.tile(N.c_[[0.5, 0., 0.001]], (1,2)))
        N.testing.assert_array_equal(dirs, N.c_[[-1., 0., 0.], [1., 0., 0.]])
    
    def test_cylinder_height(self):
        """The bounding cylinder exists for planoconvex lens"""
        f = self.lens.focal_length()
        rb = RayBundle(N.c_[[0., 0., -0.01]], N.c_[[1., 0., 0.]],
            energy=N.r_[1.], ref_index=N.r_[1.5])

        e = TracerEngine(Assembly([self.lens]))
        verts, dirs = e.ray_tracer(rb, 1, 1e-6)

        N.testing.assert_array_equal(verts, N.array([]).reshape(3,0))


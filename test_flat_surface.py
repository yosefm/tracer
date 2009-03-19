import unittest.py 
from flat_surface import FlatSurface

class TestFlatSurfaceInterface(unittest.TestCase):
    def runTest(self):
        """Doesn't allow negative width or height"""
        surf = FlatSurface()
        self.assertRaises(ValueError, surf.set_width, -42)
        self.assertRaises(ValueError, surf.set_height, -42)

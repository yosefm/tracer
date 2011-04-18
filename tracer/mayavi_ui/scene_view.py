"""
This module holds basic building blocks for displaying ray tracing scenes using
the MayaVi package of 3D plotting. A traited scene class is used to hold an
assembly and ray bundle (source), and show_assembly() and show_rays() help
with the drawing.
"""

import enthought.traits.api as t_api
import enthought.traits.ui.api as tui
from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene

import numpy as N
from ..tracer_engine import TracerEngine

class TracerScene(t_api.HasTraits):
    _scene = t_api.Instance(MlabSceneModel, ())
    
    def __init__(self, assembly, source):
        """
        TracerScene manages a MayaVi scene with one tracer assembly and one
        tracer ray bundle. It redraws the assembly whenever it is replaced, 
        and the path of the source rays from the given source bundle to their
        escape from the system.
        
        Arguments:
        assembly - the assembly to be used for tracing, an Assembly instance.
            The assembly's surfaces are drawn on the scene. All surfaces must
            provide a mesh() method.
        source - a RayBundle instance with the rays to trace.
        """
        t_api.HasTraits.__init__(self)
        self._esc = 1.
        
        self._asm = assembly
        self._source = source
        
        # Initialize drawn surfaces:
        self._meshes = show_assembly(self._scene, self._asm)
        
        # First plot:
        self._lines = []
        self.plot_ray_trace()
    
    def clear_scene(self):
        """
        Removes all objects from the MayaVi scene, in one shot, instead of
        removing elements one by one.
        """
        self._scene.mlab.clf()
        self._lines = []
        self._meshes = []
        
    def set_assembly(self, asm):
        """
        Replace the scene's assembly with a new one, and redraw.
        
        Arguments:
        asm - an Assembly instance, whose all surfacesd have a mesh() method.
        """
        self._asm = asm
        for mesh in self._meshes:
            mesh.remove()
        self._meshes = show_assembly(self._scene, self._asm)
    
    def set_source(self, source):
        """
        Replace the source, and replot the full trace from the new source.
        
        Arguments:
        source - a RayBundle instance.
        """
        self._source = source
        self.plot_ray_trace()
    
    def set_background(self, bg):
        """
        Change the background colour for the scene.
        
        Arguments:
        gb - a tuple of (R, G, B) values, each a float from 0 to 1.
        """
        self._scene.background = bg
    
    def plot_ray_trace(self):
        """
        Removes any previously plotted ray sections from the last trace of rays
        from the scene's source, retraces its rays, and plots them on the
        scene.
        """
        # Allow a scene with no source:
        if self._source is None:
            return
        
        # Remove previous rays:
        for line in self._lines:
            line.remove()
        
        # Trace new rays:
        engine = TracerEngine(self._asm)
        params = engine.ray_tracer(self._source, 20000000, .05)[0]
        self._lines = show_rays(self._scene, engine.tree, self._esc)
    
    view = tui.View(
        tui.Item('_scene', editor=SceneEditor(scene_class=MayaviScene),
            height=400, width=300, show_label=False))

def show_assembly(scene, assembly, colour=(0.5, 0.5, 0.5), resolution=10):
    """
    Add to a scene meshes for the surfaces composing an assembly
    The colour used is that given as an argument, unless a surface is
    annotated with a .color attribute, in which case this colour is used.
    The same applies to the 'resolution' argument and .resolution attribute.
    
    Arguments:
    assembly - an Assembly instance.
    colour - an R,G,B tuple; the colour it represents is applied to all
        the assembly's surfaces.
    resolution - of the meshes, in points per unit length.
    """
    meshes = []
    for surf in assembly.get_surfaces():
        if hasattr(surf, 'resolution'):
            sres = surf.resolution
        else:
            sres = resolution
        if hasattr(surf, 'colour'):
            scol = surf.colour
        else:
            scol = colour
        
        mesh = scene.mlab.mesh(*surf.mesh(sres), color=scol)
        meshes.append(mesh)
    
    return meshes

def show_rays(scene, tree, escaping_len):
    """
    Given the tree data structure from the ray tracing engine, 
    3D-plot the rays.
    
    Arguments:
    scene - an mlab sccene object on which to draw the lines.
    tree - a tree of rays, as constructed by the tracer engine
    escaping_len - the length of the arrow indicating the direction of rays
        that don't intersect any surface (leaf rays).
    
    Returns:
    lines - a list of plot3d objects, each representing a ray segment on the
        scene.
    """
    lines = []
    
    for level in xrange(len(tree)):
        start_rays = tree[level]
        sv = start_rays.get_vertices()
        sd = start_rays.get_directions()
        se = start_rays.get_energy()
        
        if level == len(tree) - 1:
            parents = []
        else:
            end_rays = tree[level + 1]
            ev = end_rays.get_vertices()
            parents = end_rays.get_parents()

        for ray in xrange(start_rays.get_num_rays()):
            if se[ray] == 0:
                continue
            
            if ray in parents:
                # Has a hit on another surface
                first_child = N.where(ray == parents)[0][0]
                endpoints = N.c_[sv[:,ray], ev[:,first_child]]
            else:
                # Escaping ray.
                endpoints = N.c_[sv[:,ray], sv[:,ray] + sd[:,ray]*escaping_len]
            
            lines.append(scene.mlab.plot3d(*endpoints, tube_radius=None))
    tree.pop()
    return lines


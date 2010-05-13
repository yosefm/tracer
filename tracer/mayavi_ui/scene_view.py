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
        Argumennts:
        assembly - the assembly to be used for tracing, an Assembly instance.
            The assembly's surfaces are drawn on the scene.
        source - a RayBundle instance with the rays to trace.
        """
        t_api.HasTraits.__init__(self)
        
        self._asm = assembly
        self._source = source
        
        # Initialize drawn surfaces:
        show_assembly(self._scene, self._asm)
        
        # First plot:
        self._lines = []
        self.plot_ray_trace()
    
    def plot_ray_trace(self):
        """
        Removes any previously plotted ray sections from the last trace of rays
        from the scene's source, retraces its rays, and plots them on the
        scene.
        """
        # Remove previous rays:
        for line in self._lines:
            line.remove()
        
        # Trace new rays:
        engine = TracerEngine(self._asm)
        params = engine.ray_tracer(self._source, 20000000, .05)[0]
        self._lines = show_rays(self._scene, engine.tree, 1)
    
    view = tui.View(
        tui.Item('_scene', editor=SceneEditor(scene_class=MayaviScene),
            height=400, width=300, show_label=False))

def show_assembly(scene, assembly, colour=(0.5, 0.5, 0.5), resolution=0.1):
    """
    Add to a scene meshes for the surfaces composing an assembly
    
    Arguments:
    assembly - an Assembly instance.
    colour - an R,G,B tuple; the colour it represents is applied to all
        the assembly's surfaces.
    resolution - of the meshes, in points per unit length.
    """
    meshes = []
    for surf in assembly.get_surfaces():
        mesh = scene.mlab.mesh(*surf.mesh(resolution), color=colour)
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
        if level == len(tree) - 1:
            parents = []
        else:
            end_rays = tree[level + 1]
            parents = end_rays.get_parent()

        for ray in xrange(start_rays.get_num_rays()):
            if ray in parents:
                # Has a hit on another surface
                first_child = N.where(ray == parents)[0][0]
                endpoints = N.c_[
                    start_rays.get_vertices()[:,ray],
                    end_rays.get_vertices()[:,first_child] ]
            else:
                # Escaping ray.
                endpoints = N.c_[
                    start_rays.get_vertices()[:,ray],
                    start_rays.get_vertices()[:,ray] + \
                        start_rays.get_directions()[:,ray]*escaping_len ]
            
            lines.append(scene.mlab.plot3d(*endpoints))
    tree.pop()
    return lines


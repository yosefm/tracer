"""
This module holds basic building blocks for displaying ray tracing scenes using
the MayaVi package of 3D plotting. A traited scene class is used to hold an
assembly and ray bundle (source), and show_assembly() and show_rays() help
with the drawing.
"""

import traits.api as t_api
import traitsui.api as tui
from mayavi.tools.mlab_scene_model import MlabSceneModel
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.core.ui.mayavi_scene import MayaviScene

import numpy as N
from ..tracer_engine import TracerEngine

class TracerScene(t_api.HasTraits):
    _scene = t_api.Instance(MlabSceneModel, ())
    
    def __init__(self, assembly, source, escaping=1., ray_selector=None):
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
        self._esc = escaping
        
        self.set_assembly(assembly)
        self._source = source
        
        # Trace control:
        self._num_trace_iters = 200000000 # default to infinity, pretty much
        self._min_energy = 0.05
        
        # First plot:
        self._lines = None
        self._ray_selector = ray_selector
        self.plot_ray_trace()
        
        # Default view:
        self.view = tui.View(self.scene_view_item())
    
    def clear_scene(self):
        """
        Removes all objects from the MayaVi scene, in one shot, instead of
        removing elements one by one.
        """
        self._scene.mlab.clf()
        self._lines = None
        for surf_id, mapping in self._meshes.iteritems():
            if mapping[1] is not None:
                mapping[1].remove()
            self._meshes[surf_id] = (mapping[0], None)
        
    def set_assembly(self, asm):
        """
        Replace the scene's assembly with a new one, and redraw.
        
        Arguments:
        asm - an Assembly instance, whose all surfacesd have a mesh() method.
        """
        self._asm = asm
        self._meshes = dict((id(s), (s, None)) for s in asm.get_surfaces())
        self.show_assembly()
    
    def update_surfaces(self):
        """
        Recheck the list of surfaces belonging to the assembly, removing from
        view surfaces that were deleted from the assembly, and adding to it
        surfaces that weren't present before. Does not change existing
        surfaces - use show_assembly(update=...) for that.
        """
        current_surfs = self._asm.get_surfaces()
        surf_ids = [id(s) for s in current_surfs]
        
        # Remove:
        for sid in self._meshes.keys():
            if sid not in surf_ids:
                self._meshes[sid][1].remove()
                del self._meshes[sid]
        
        # Add:
        new_surfs = []
        for six, sid in enumerate(surf_ids):
            if sid not in self._meshes:
                self._meshes[sid] = (current_surfs[six], None)
                new_surfs.append(sid)
        self.show_assembly(update=new_surfs)
    
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
    
    def trace_control(self):
        """
        Returns a tuple (max. iterations, min. energy) used in ray tracing.
        """
        return self._num_trace_iters, self._min_energy
    
    def set_trace_control(self, **kwds):
        """
        Sets various ray tracing parameters. Currently supported keywords:
        max_iter - maximum number of trace iterations.
        min_energy - discontinue rays with energy lower than this.
        """
        if 'max_iter' in kwds:
            self._num_trace_iters = kwds['max_iter']
        if 'min_energy' in kwds:
            self._min_energy = kwds['min_energy']
    
    def plot_ray_trace(self):
        """
        Removes any previously plotted ray sections from the last trace of rays
        from the scene's source, retraces its rays, and plots them on the
        scene.
        """
        # Allow a scene with no source:
        if self._source is None:
            return
        
        self._scene.disable_render = True
        
        # Remove previous rays:
        if self._lines is not None:
            self._lines.remove()
            self._lines = None
        
        # Trace new rays:
        engine = TracerEngine(self._asm)
        params = engine.ray_tracer(self._source, 
            self._num_trace_iters, self._min_energy)[0]
        self._lines = show_rays(self._scene, engine.tree, self._esc, 
            self._ray_selector)
        
        self._scene.disable_render = False
    
    @staticmethod
    def scene_view_item(height=400, width=300):
        """
        Generates an item placable on TraitsUI views, including all necessary
        imports, so that not every non-trivial usage requires tons of imports.
        
        Arguments:
        height, width - of the tui.Item, passed directly to the constructor.
        """
        return tui.Item('_scene', editor=SceneEditor(scene_class=MayaviScene),
            height=height, width=width, show_label=False)

    def show_assembly(self, colour=(0.5, 0.5, 0.5), resolution=10,
        update=None):
        """
        Add to a scene meshes for the surfaces composing an assembly
        The colour used is that given as an argument, unless a surface is
        annotated with a .color attribute, in which case this colour is used.
        The same applies to the 'resolution' argument and .resolution attribute.
        
        Arguments:
        colour - an R,G,B tuple; the colour it represents is applied to all
            the assembly's surfaces.
        resolution - of the meshes, in points per unit length.
        update - a list of surface ids (Python function id()) to update.
            If None, all surfaces are updated. For adding or removing surfaces,
            use set_assembly().
        """
        for surf_id, mapping in self._meshes.iteritems():
            if update is not None and surf_id not in update:
                continue
            surf, mesh = mapping
            
            # Set visual properties:
            if hasattr(surf, 'resolution'):
                sres = surf.resolution
            else:
                sres = resolution
            if hasattr(surf, 'colour'):
                scol = surf.colour
            else:
                scol = colour
            
            # Delete existing mesh:
            if mesh is not None:
                mesh.remove()
            
            # Replace:
            mesh = self._scene.mlab.mesh(*surf.mesh(sres), color=scol)
            self._meshes[surf_id] = (surf, mesh)

def show_rays(scene, tree, escaping_len, ray_selector=None):
    """
    Given the tree data structure from the ray tracing engine, 
    3D-plot the rays.
    
    Arguments:
    scene - an mlab sccene object on which to draw the lines.
    tree - a tree of rays, as constructed by the tracer engine
    escaping_len - the length of the arrow indicating the direction of rays
        that don't intersect any surface (leaf rays).
    ray_selector (optional) - a function that takes a ray bundle and returns a
        boolean array stating whether to show each ray.
    
    Returns:
    lines - a list of plot3d objects, each representing a ray segment on the
        scene.
    """
    points = [tree[0].get_vertices()]
    accum_points = [0, tree[0].get_num_rays()]
    connections = []
    
    for level in xrange(tree.num_bunds()):
        start_rays = tree[level]
        se = start_rays.get_energy()
        
        non_degenerate = se != 0
        if ray_selector is not None:
            non_degenerate &= ray_selector(start_rays)
        
        sv = start_rays.get_vertices()[:,non_degenerate]
        sd = start_rays.get_directions()[:,non_degenerate]
        
        if level == tree.num_bunds() - 1:
            # Make endpoints for escaping rays
            ev = sv + sd*escaping_len
            parents = N.arange(ev.shape[1])
        else:
            end_rays = tree[level + 1]
            escaping = ~N.any(
                N.nonzero(non_degenerate)[0] == end_rays.get_parents()[:,None], 0)
            escaping_endpoints = sv[:,escaping] + sd[:,escaping]*escaping_len
            
            ev = N.hstack((end_rays.get_vertices(), escaping_endpoints))
            parents = N.hstack((end_rays.get_parents(), N.nonzero(escaping)[0]))
            
        points.append(ev)
        accum_points.append(accum_points[-1] + ev.shape[1])
        
        connections.append(N.hstack((accum_points[level] + parents[:,None],
            accum_points[level + 1] + N.arange(ev.shape[1])[:,None] )) )
    
    points = N.hstack(points)
    connections = N.vstack(connections)
    lines = scene.mlab.pipeline.scalar_scatter(*points)
    lines.mlab_source.dataset.lines = connections
    lines = scene.mlab.pipeline.surface(lines, line_width=1)
    return lines


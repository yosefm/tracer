import enthought.traits.api as t_api
import enthought.traits.ui.api as tui
from enthought.mayavi.tools.mlab_scene_model import MlabSceneModel
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene

import numpy as N

from ..models.one_sided_mirror import rect_one_sided_mirror
from ..assembly import Assembly
from ..ray_bundle import RayBundle
from ..tracer_engine import TracerEngine
from .. import spatial_geometry as G

class ExampleScene(t_api.HasTraits):
    scene = t_api.Instance(MlabSceneModel, ())
    source_y = t_api.Range(0., 5., 2.)
    source_z = t_api.Range(0., 5., 1.)
    
    def __init__(self):
        t_api.HasTraits.__init__(self)
        
        # Initialize surfaces:
        x, y = N.mgrid[-5:5.1, -5:5.1]
        z = N.zeros_like(x)
        self.upright_surf = self.scene.mlab.mesh(x, y, z, color=(1,0,0))
        self.slanted_surf = self.scene.mlab.mesh(x, y/N.sqrt(2), y/N.sqrt(2), color=(0,0,1))
        
        # The energy bundle we'll use for now:
        nrm = 1/(N.sqrt(2))
        direct = N.c_[[0,-nrm, nrm],[0,0,-1]]
        position = N.tile(N.c_[[0, self.source_y, self.source_z]], (1, 2))
        self.bund = RayBundle(vertices=position, directions=direct, energy=N.r_[1, 1])
        
        # The assembly for ray tracing:
        rot1 = G.rotx(N.pi/4)[:3,:3]
        surf1 = rect_one_sided_mirror(width=10, height=10)
        surf1.set_rotation(rot1)
        surf2 = rect_one_sided_mirror(width=10, height=10)
        self.assembly = Assembly(objects=[surf1, surf2])
        
        # First plot:
        self.lines = []
        self.plot_ray_trace()
    
    @t_api.on_trait_change('scene.activated')
    def initialize_camere(self):
        self.scene.mlab.view(0, -90)
        self.scene.mlab.roll(0)
    
    @t_api.on_trait_change('source_y, source_z')
    def bundle_move(self):
        position = N.tile(N.c_[[0, self.source_y, self.source_z]], (1, 2))
        self.bund.set_vertices(position)
        self.plot_ray_trace()

    view = tui.View(
        tui.Item('scene', editor=SceneEditor(scene_class=MayaviScene),
            height=400, width=300, show_label=False),
        tui.HGroup('-', 'source_y', 'source_z'))
    
    def plot_ray_trace(self):
        # Remove previous rays:
        for line in self.lines:
            line.remove()
        
        # Trace new rays:
        engine = TracerEngine(self.assembly)
        params = engine.ray_tracer(self.bund, 5, .05)[0]
        self.lines = self.show_rays(engine.tree, 1)

    def show_rays(self, tree, escaping_len):
        """
        Given the tree data structure from the ray tracing engine, 
        3D-plot the rays.
        """
        lines = []
        # Add an empty level to the tree just to have a simpler loop:
        tree.append(RayBundle.empty_bund())
        
        for level in xrange(len(tree) - 1):
            start_rays = tree[level]
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
                
                lines.append(self.scene.mlab.plot3d(*endpoints))
        tree.pop()
        return lines

def app():
    scene = ExampleScene()
    scene.configure_traits()


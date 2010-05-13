"""
This example uses the MayaVi2 package and the mayavi_ui subpackage to show
a scene with two flat surfaces and two rays reflected off them. You can play
with the location of the source of the rays.
"""

import enthought.traits.api as t_api
import enthought.traits.ui.api as tui
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene

from tracer.mayavi_ui.scene_view import TracerScene

import numpy as N
from tracer.models.one_sided_mirror import rect_one_sided_mirror
from tracer.assembly import Assembly
from tracer.ray_bundle import RayBundle
from tracer import spatial_geometry as G

class ExampleScene(TracerScene):
    source_y = t_api.Range(0., 5., 2.)
    source_z = t_api.Range(0., 5., 1.)

    def __init__(self):
        # The energy bundle we'll use for now:
        nrm = 1/(N.sqrt(2))
        direct = N.c_[[0,-nrm, nrm],[0,0,-1]]
        position = N.tile(N.c_[[0, self.source_y, self.source_z]], (1, 2))
        self.bund = RayBundle(vertices=position, directions=direct, energy=N.r_[1, 1])

        # The assembly for ray tracing:
        rot1 = N.dot(G.rotx(N.pi/4)[:3,:3], G.roty(N.pi)[:3,:3])
        surf1 = rect_one_sided_mirror(width=10, height=10)
        surf1.set_rotation(rot1)
        surf2 = rect_one_sided_mirror(width=10, height=10)
        self.assembly = Assembly(objects=[surf1, surf2])

        TracerScene.__init__(self, self.assembly, self.bund)

    @t_api.on_trait_change('_scene.activated')
    def initialize_camere(self):
        self._scene.mlab.view(0, -90)
        self._scene.mlab.roll(0)

    @t_api.on_trait_change('source_y, source_z')
    def bundle_move(self):
        position = N.tile(N.c_[[0, self.source_y, self.source_z]], (1, 2))
        self.bund.set_vertices(position)
        self.plot_ray_trace()

    view = tui.View(
        tui.Item('_scene', editor=SceneEditor(scene_class=MayaviScene),
            height=400, width=300, show_label=False),
        tui.HGroup('-', 'source_y', 'source_z'))

if __name__ == '__main__':
    scene = ExampleScene()
    scene.configure_traits()


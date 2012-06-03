"""
This example shows, in 3D,  a dish with a homogenizer, defined by its geometric
concentration, and number of reflections in the homogenizer. You can play with
both values, and each change will cause the scene to redraw.
"""

import traits.api as t_api
import traitsui.api as tui
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.core.ui.mayavi_scene import MayaviScene

from tracer.mayavi_ui.scene_view import TracerScene

import numpy as N
from tracer.sources import solar_disk_bundle
from tracer.models.tau_minidish import standard_minidish
from tracer import spatial_geometry as G

class DishScene(TracerScene):
    """
    Extends TracerScene with the variables required for this example and adds
    handling of simulation-specific details, like colouring the dish elements
    and setting proper resolution.
    """
    refl = t_api.Float(1., label='Edge reflections')
    concent = t_api.Float(450, label='Concentration')
    disp_num_rays = t_api.Int(10)
    
    def __init__(self):
        dish, source = self.create_dish_source()
        TracerScene.__init__(self, dish, source)
        self.set_background((0., 0.5, 1.))
    
    def create_dish_source(self):
        """
        Creates the two basic elements of this simulation: the parabolic dish,
        and the pillbox-sunshape ray bundle. Uses the variables set by 
        TraitsUI.
        """
        dish, f, W, H = standard_minidish(1., self.concent, self.refl, 1., 1.)
        # Add GUI annotations to the dish assembly:
        for surf in dish.get_homogenizer().get_surfaces():
            surf.colour = (1., 0., 0.)
        dish.get_main_reflector().colour = (0., 0., 1.)

        source = solar_disk_bundle(self.disp_num_rays,
            N.c_[[0., 0., f + H + 0.5]], N.r_[0., 0., -1.], 0.5, 0.00465)
        source.set_energy(N.ones(self.disp_num_rays)*1000./self.disp_num_rays)
        
        return dish, source

    @t_api.on_trait_change('refl, concent')
    def recreate_dish(self):
        """
        Makes sure that the scene is redrawn upon dish design changes.
        """
        dish, source = self.create_dish_source()
        self.set_assembly(dish)
        self.set_source(source)
    
    # Parameters of the form that is shown to the user:
    view = tui.View(
        tui.Item('_scene', editor=SceneEditor(scene_class=MayaviScene),
            height=500, width=500, show_label=False),
        tui.HGroup('-', 
            tui.Item('concent', editor=tui.TextEditor(evaluate=float, auto_set=False)), 
            tui.Item('refl', editor=tui.TextEditor(evaluate=float, auto_set=False))))


if __name__ == '__main__':
    scene = DishScene()
    scene.configure_traits()


"""
Yet another MayaVi example: a heliostat field.
"""

import enthought.traits.api as t_api
import enthought.traits.ui.api as tui
from enthought.tvtk.pyface.scene_editor import SceneEditor
from enthought.mayavi.core.ui.mayavi_scene import MayaviScene

from tracer.mayavi_ui.scene_view import TracerScene

import numpy as N
from scipy.constants import degree

from tracer.ray_bundle import RayBundle
from tracer.assembly import Assembly
from tracer.spatial_geometry import roty

from tracer.models.one_sided_mirror import one_sided_receiver
from tracer.models.heliostat_field import HeliostatField, radial_stagger

class TowerScene(TracerScene):
    sun_az = t_api.Range(0, 180, 90, label="Sun azimuth")
    sun_elev = t_api.Range(0, 90, 45, label="Sun elevation")
    
    def __init__(self):
        xy = radial_stagger(-N.pi/4, N.pi/4 + 0.0001, N.pi/8, 5, 20, 1)
        self.pos = N.hstack((xy, N.zeros((xy.shape[0], 1))))
        
        self.field = HeliostatField(self.pos, 0.5, 0.5, 0, 10)
        rec = one_sided_receiver(1., 1.)[1]
        rec_trans = roty(N.pi/2)
        rec_trans[2,3] = 10
        rec.set_transform(rec_trans)
        self.plant = Assembly(objects=[rec], subassemblies=[self.field])
        TracerScene.__init__(self, self.plant, self.gen_rays())
        self.aim_field()
        self.set_background((0., 0.5, 1.))
    
    def gen_rays(self):
        sun_z = N.cos(self.sun_elev*degree)
        sun_xy = N.r_[-N.sin(self.sun_az*degree), -N.cos(self.sun_az*degree)]
        sun_vec = N.r_[sun_xy*N.sqrt(1 - sun_z**2), sun_z]
        
        rpos = (self.pos + sun_vec).T
        direct = N.tile(-sun_vec, (self.pos.shape[0], 1)).T
        rays = RayBundle(rpos, direct, energy=N.ones(self.pos.shape[0]))
        
        return rays
    
    @t_api.on_trait_change('sun_az, sun_elev')
    def aim_field(self):
        self.clear_scene()
        rays = self.gen_rays()
        self.field.aim_to_sun(self.sun_az*degree, self.sun_elev*degree)
        
        self.set_source(rays)
        self.set_assembly(self.plant) # Q&D example.
    
    @t_api.on_trait_change('_scene.activated')
    def initialize_camere(self):
        self._scene.mlab.view(0, -90)
        self._scene.mlab.roll(90)
    
    # Parameters of the form that is shown to the user:
    view = tui.View(
        tui.Item('_scene', editor=SceneEditor(scene_class=MayaviScene),
            height=500, width=500, show_label=False),
        tui.HGroup('-', 'sun_az', 'sun_elev'))


if __name__ == '__main__':
    scene = TowerScene()
    scene.configure_traits()


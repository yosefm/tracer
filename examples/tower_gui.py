"""
Yet another MayaVi example: a heliostat field. In this example we also use an
embedded Matplotlib figure to show the flux map on request.
"""

import traits.api as t_api
import traitsui.api as tui

from tracer.mayavi_ui.scene_view import TracerScene

import numpy as N
from scipy.constants import degree

from tracer.ray_bundle import RayBundle
from tracer.sources import pillbox_sunshape_directions
from tracer.assembly import Assembly
from tracer.spatial_geometry import roty, rotation_to_z
from tracer.tracer_engine import TracerEngine

from tracer.models.one_sided_mirror import one_sided_receiver
from tracer.models.heliostat_field import HeliostatField, radial_stagger, solar_vector

# For the embedded flux map:
from matplotlib.figure import Figure 
from embedded_figure import MPLFigureEditor
import wx

class TowerScene(TracerScene):
    # Location of the sun:
    sun_az = t_api.Range(0, 180, 90, label="Sun azimuth")
    sun_elev = t_api.Range(0, 90, 45, label="Sun elevation")
    
    # Heliostat placement distance:
    radial_res = t_api.Float(1., label="Radial distance")
    ang_res = t_api.Float(N.pi/8, lable="Angular distance")
    
    # Flux map figure:
    fmap = t_api.Instance(Figure)
    fmap_btn = t_api.Button(label="Update flux map")
    
    def __init__(self):
        self.gen_plant()
        TracerScene.__init__(self, self.plant, self.gen_rays())
        
        self.aim_field()
        self.set_background((0., 0.5, 1.))
    
    def gen_rays(self):
        sun_vec = solar_vector(self.sun_az*degree, self.sun_elev*degree)
        rpos = (self.pos + sun_vec).T
        direct = N.tile(-sun_vec, (self.pos.shape[0], 1)).T
        rays = RayBundle(rpos, direct, energy=N.ones(self.pos.shape[0]))
        
        return rays
    
    def gen_plant(self):
        xy = radial_stagger(-N.pi/4, N.pi/4 + 0.0001, self.ang_res, 5, 20, self.radial_res)
        self.pos = N.hstack((xy, N.zeros((xy.shape[0], 1))))
        self.field = HeliostatField(self.pos, 0.5, 0.5, 0, 10)

        self.rec, recobj = one_sided_receiver(1., 1.)
        rec_trans = roty(N.pi/2)
        rec_trans[2,3] = 10
        recobj.set_transform(rec_trans)

        self.plant = Assembly(objects=[recobj], subassemblies=[self.field])
    
    @t_api.on_trait_change('sun_az, sun_elev')
    def aim_field(self):
        self.clear_scene()
        rays = self.gen_rays()
        self.field.aim_to_sun(self.sun_az*degree, self.sun_elev*degree)
        
        self.set_assembly(self.plant) # Q&D example.
        self.set_source(rays)
    
    @t_api.on_trait_change('radial_res, ang_res')
    def replace_plant(self):
        self.gen_plant()
        self.aim_field()
    
    @t_api.on_trait_change('_scene.activated')
    def initialize_camere(self):
        self._scene.mlab.view(0, -90)
        self._scene.mlab.roll(90)
    
    def _fmap_btn_fired(self):
        """Generate a flux map using much more rays than drawn"""
        # Generate a large ray bundle using a radial stagger much denser
        # than the field.
        sun_vec = solar_vector(self.sun_az*degree, self.sun_elev*degree)
        
        hstat_rays = 1000
        num_rays = hstat_rays*len(self.field.get_heliostats())
        rot_sun = rotation_to_z(-sun_vec)
        direct = N.dot(rot_sun, pillbox_sunshape_directions(num_rays, 0.00465))
        
        xy = N.random.uniform(low=-0.25, high=0.25, size=(2, num_rays))
        base_pos = N.tile(self.pos, (hstat_rays, 1)).T
        base_pos += N.dot(rot_sun[:,:2], xy)
        
        base_pos -= direct
        rays = RayBundle(base_pos, direct, energy=N.ones(num_rays))
        
        # Perform the trace:
        self.rec.get_optics_manager().reset()
        e = TracerEngine(self.plant)
        e.ray_tracer(rays, 1000, 0.05)
        
        # Show a histogram of hits:
        energy, pts = self.rec.get_optics_manager().get_all_hits()
        x, y = self.rec.global_to_local(pts)[:2]
        rngx = 0.5
        rngy = 0.5
        
        bins = 50
        H, xbins, ybins = N.histogram2d(x, y, bins, \
            range=([-rngx,rngx], [-rngy,rngy]), weights=energy)
        
        self.fmap.axes[0].images=[]
        self.fmap.axes[0].imshow(H, aspect='auto')
        wx.CallAfter(self.fmap.canvas.draw) 
    
    def _fmap_default(self):
        figure = Figure()
        figure.add_axes([0.05, 0.04, 0.9, 0.92])
        return figure
    
    # Parameters of the form that is shown to the user:
    view = tui.View(tui.HGroup(tui.VGroup(
        TracerScene.scene_view_item(500, 500),
        tui.HGroup('-', 'sun_az', 'sun_elev'),
        tui.HGroup('radial_res', 'ang_res'),
        tui.Item('fmap_btn', show_label=False)),
        tui.Item('fmap', show_label=False, editor=MPLFigureEditor())))


if __name__ == '__main__':
    scene = TowerScene()
    scene.configure_traits()


# Implements a surface specifically as a recieving surface                                 

import numpy as N
import pylab as P
from flat_surface import FlatSurface
from ray_bundle import RayBundle
import optics

class Receiver(FlatSurface):
    """                                                                                 
  Implements a flat recieving surface for the rays   
    """
    def __init__(self, location=None, rotation=None, absorptivity=1.,width=1.,height=1.):
        FlatSurface.__init__(self, location, rotation, absorptivity, width, height)
        self.coordinates = []
        self.energy = []

    def get_outgoing(self, selector, energy, parent):
        vertices = N.dot(self.get_rotation()[:, :2],  self._current_params[:, selector]) + \
            self.get_location()[:, None]
        dirs = optics.reflections(self._current_bundle.get_directions()[:, selector],
            self.get_rotation()[:, 2][:,None])
        outg = RayBundle()
        outg.set_vertices(vertices)
        outg.set_directions(dirs)
        outg.set_energy(energy[:,selector])
        new_parent = parent[selector]
        outg.set_parent(new_parent)
        self.collect_energy(outg)
        return outg

    def collect_energy(self, bundle):
        """                                                                                 
        Saves the values of the coordinates and energy of incoming rays                     
        """
        self.coordinates.append(bundle.get_vertices())
        self.energy.append(bundle.get_energy())
  
    def plot_energy(self):
        """                                                                                 
        Plots the energy distribution on the receiving surface                              
        """
        coords = self.coordinates[0]
        coords_rot = N.dot(self.get_rotation(), coords)
        energy = N.array(self.energy)
        x = coords_rot[0]  # this should be by row is there is more than one
        y = coords_rot[1]  # receiving surface; also this is the local x, y
                
        P.scatter(x, y)
        P.show()


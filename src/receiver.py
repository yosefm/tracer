# Implements a surface specifically as a recieving surface                                 

import numpy as N
import pylab as P
from flat_surface import FlatSurface
from ray_bundle import RayBundle
import optics

class Receiver(FlatSurface):
    """                                                                                 
  Implements a flat recieving surface for the rays
  Private attributes:
  _coordinates - a list of the coordinates of where a ray intersected with the surface
  _energy - a list of the energy corresponding to the intersecting rays
    """
    def __init__(self, location=None, rotation=None, absorptivity=1.,width=1.,height=1.):
        FlatSurface.__init__(self, location, rotation, absorptivity, width, height)
        self._coordinates = []
        self._energy = []

    def get_outgoing(self, selector):
        """
        Gets the outgoing ray as it would for any surface, but also calls the 
        collect_energy() function to then store the coordinates and energy of the
        points of intersection on the receiving surface
        Returns: the outgoing ray 
        """
        outg = FlatSurface.get_outgoing(self,selector)
        self.collect_energy(outg) 
        return outg

    def collect_energy(self, bundle):
        """                                                                                 
        Saves the values of the coordinates and energy of incoming rays                     
        """
        self._coordinates.append(bundle.get_vertices())
        self._energy.append(bundle.get_energy())
  
    def plot_energy(self):
        """                                                                                 
        Plots the energy distribution on the receiving surface                              
        """
        coords = self._coordinates[0]
        coords_rot = N.dot(self.get_rotation(), coords)
#        rot = N.array([[1,1,0],[0,.707,-.707],[0,.707,.707]])
#        coords_rot = N.dot(rot, coords)
        energy = N.array(self._energy)
    
        x = coords_rot[0]  # this should be by row is there is more than one
        y = coords_rot[1]  # receiving surface; also this is the local x, y
        
        P.scatter(x, y)
        P.show()


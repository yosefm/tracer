# A class that stores all ray bundles throughout a trace, and can answer
# queries about the path each ray took through the trace.

import numpy as np

class RayTree(object):
    def __init__(self):
        self._bunds = []
        
    def __getitem__(self, level):
        return self._bunds[level]
    
    def num_bunds(self):
        return len(self._bunds)
    
    def append(self, bund):
        """
        Add a new bundle to the trace, Parent indices are adjusted so that 
        min_energy rays are not forgotten, because tracer_engine registers
        parents based on the diluted bundle.
        
        Arguments:
        bundle - the latest RayBundle that the trace generated.
        """
        self._bunds.append(bund)
    
    def ordered_parents(self):
        """
        Returns a list of parent arrays in trace order (from first hit of source bundle to 
        last bundle). 
        """
        return [bund.get_parents() for bund in self._bunds[1:]]
    
    def ray_history(ray_index, level=None):
        """
        Return, in reverse order, all the indices into ray bundles starting
        from the given ray, going back to the original bundle. Starts with the
        given ray_index.
        
        Arguments:
        ray_index - the tracked ray's index at its bundle.
        level - the number of bundle to go back from (the source bundle is 0).
            defaults to last bundle.
        """
        if level is None:
            level = self.num_bunds()
        
        parents = np.empty(level)
        parents[0] = ray_index
        
        for pix in xrange(1, level):
            parents[pix] = \
                self._bunds[level - pix + 1].get_parents()[parents[pix - 1]]
        
        return parents


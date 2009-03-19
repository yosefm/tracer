import numpy as N
import pylab as P

import ray_bundle
from flat_surface import FlatSurface

dir = N.array([0., 0, -1])
center = N.array([0,  0, 2]).reshape(-1, 1)
bund = ray_bundle.solar_disk_bundle(5000,  center,  dir,  2,  N.pi/1000.)

surf = FlatSurface()
inters = ~N.isinf(surf.register_incoming(bund))

v = bund.get_vertices()[:, ~inters]
d = bund.get_directions()[:, ~inters]
P.quiver(v[0], v[1], d[0], d[1], scale=0.1)

outg = surf.get_outgoing(inters)
v = outg.get_vertices()
d = outg.get_directions()
P.quiver(v[0], v[1], d[0], d[1],  scale=0.2)

P.show()

import numpy as N
import pylab as P

import ray_bundle
from flat_surface import FlatSurface

# Create ray bundle
dir = N.array([0., 0, -1])
center = N.array([0,  0, 2]).reshape(-1, 1)
bund = ray_bundle.solar_disk_bundle(5000,  center,  dir,  1,  N.pi/1000.)

# Intersect the bundle with a flat surface
surf = FlatSurface()
inters = ~N.isinf(surf.register_incoming(bund))

# Show non-intersecting rays
v = bund.get_vertices()[:, ~inters]
d = bund.get_directions()[:, ~inters]
P.quiver(v[0], v[1], d[0], d[1], scale=0.1)

# Show returning rays.
outg = surf.get_outgoing(inters)
v = outg.get_vertices()
d = outg.get_directions()
P.quiver(v[0], v[1], d[0], d[1],  scale=0.2, color='red')

P.show()

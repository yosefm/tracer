
Tracer-surface communication protocol
-------------------------------------

The communication between the tracer engine and the surfaces happens in the
following stages:

1. The tracer calls a function register_incoming() on a surface, passing in a ray
   bundle object. In return, the surface gives an array where for each incoming ray
   the parametric position along the ray where the ray hits is returned; rays that
   don't hit the surface are marked with infinity.

2. The engine decides, for each ray, which of the surfaces in the scene has the
   closest intersection with that ray; it then informs the surface that only these
   rays will be queried next, using Surface.select_rays().

3. The engine calls each surface's get_outgoing() routine, with a selector
   specifying which of the incoming rays to answer about. The surface is expected to
   already have all the information needed for the answer. The surface replies with a
   new RayBundle object which may include reflected and refracted rays.

4. The engine then takes all the answers, welds them into a new ray bundle, and
   uses that bundle for the next iteration.

* Either after 3 or, if no rays intersected a surface after 1, the ray tracer
  calls the surface's done() method to allow it to clear memory.


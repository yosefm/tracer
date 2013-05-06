Some design notes
=================

This document is meant to explain a bit about the design of the tracer, its 
components and goals.

Overview
--------
The tracer is concerned with a scene in which light rays can 
move about freely. In the scene we have objects that rays may interact with. A ray 
in free space may interact with an object by intersecting with its surface, at 
which point it can be reflected or refracted.

Tracing a ray is a question of selecting which surface is intersected by which 
rays, and at what point along the ray. This is the purpose of the TracerEngine 
class; Surface subclasses interact with the tracer engine through a communication 
protocol that the surfaces are expected to follow.


Communication protocol
----------------------
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


Energy bundles
--------------
A ray-tracing algorithm cab work with two different kinds of rays: ones with
discrete amounts of energy ("energy bundles") and ones with continuous
(floating-point) amounts of energy. For the first case, statistical rules are
used to determine absorption, reflectance, etc. This software uses the second
method.

Working with continuous energy, each ray's intersection with a surface may
cause several new rays to be created, according to the laws of reflection and
refraction, and given Fresnel's formulae (only specular reflections are
currently supported). A resulting ray may be discarded if it carries energy
lower than a user-defined threshold.


Surface culling
---------------
It is up to enclosed objects to determine which of their surfaces are relevant
for future intersection finding. Given a set of intersections, get_outgoing()
is when the surface calculates the outgoing rays, at which point it is possible
to know which rays will not intersect surfaces in this object at the next
iteration. Alternatively, it is possible to know, as in a lens, that only this
object needs to be checked at the next iteration.

So, for each ray we need two answers: a list of relevant surfaces; and a
boolean "ownership" flag stating that the object takes ownership for the next
iteration.

* One-sided mirror: simple. Never owns, both surfaces irrelevant after next
  iteration.
* Lens: refracted rays owned, reflected rays disowned, for rays coming from the
  outside. Opposite for rays moving inside the lens. For rays going in, all
  surfaces are relevant except (perhaps) the starting surfaces, and opposite
  for rays going outside.

Who knows what: 
* object knows what policy to take
* surface knows what was reflected or refracted and to which side
* object knows which refractive index to expect.

So: a general lens needs to have cooperative surfaces. Possibility: change
RefractiveHomogenous to mark the rays in the RayBundle instance as refracted/
reflected. Maybe change all GMs to mark the side of outgoing rays, so if the
object knows which side is which (and it should), we're independent of
refractive index.

A simple homogenous lens only requires the refractive index of the ray,
although the more general solution won't suffer from index-matching issues.

A reflective object requires no co-operation.

A default object: owns nothing, all surfaces relevant.
One-sided mirror: own nothing, all surfaces irrelevant.


Meshes
------
In the current architecture, all knowledge about a surface is encapsualted in
the Surface object or its geometry manager and optics manager. The only exposed
information about it is its transform. It might be desired by some external
tool to handle assemblies in a way that requires some knowledge about the
surfaces, without having that knowledge in advance; off the top of my head I
can think of scene graph manipulation, volume and extent calculations, or (the
use case that prompted this) drawing the assembly in a UI.

Since future uses are hard to predict, meshes are a tool that gives some very
general information about a surface, from which a lot of information can be
gleaned. Therefore, surfaces will have a mesh() method that provides a mesh
representing the object in a given resolution. 


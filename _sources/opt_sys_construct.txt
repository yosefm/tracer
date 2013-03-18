
Constructing an Optical System
===============================

Each optical system is composed of a hierarchy consisting of at least three
levels. The lowest is the surface level, responsible for finding ray
intersections and generating a ray bundle transformed from the incoming bundle
using optical laws. Surfaces are contained in optical objects, which may
provide information about dependence or relationships between surfaces. 
Finally, an assembly object contains objects and/or other assemblies.

Spatial Representation
----------------------

Each object is represented by a **frame** - a representation of its axes
relative to the object containing it. For example, a surface's frame represents
its axes in the optical object's axes; the top-level assembly has a frame
representing its axes in the "world" coordinate system. By convention, this
system is represented in local coordinates, i.e. the Z axis is up (normal to
the ground), the X axis points north, and the Y axis west.

The Surface and Assembly class are subclasses of the HasFrame class which keeps
track of the frame. the AssembledObject (optical object) class is a subclass of
Assembly, and hence of HasFrame.

Each HasFrame object has its own frame and a temporary frame. The temporary
frame is updated whenever the object's frame or one of its parents' frames
changes, and it reflects the object's frame relative to the world coordinates.

See also :doc:`has_frame`

Assemblies
----------
An assembly is just a way to hold together parts of the optical system. Aside
from having its own frame, it is also responsible for informing its
constituents of transformation changes relative to the world coordinates.

As example of usage, say you want to build a concentrating system with a
homogenizer component close to the focus. You create an assembly for your
concentrator, then add to it the assembly receiver from 
``tracer.models.homogenizer.rect_homogenizer()``; you set this sub-assembly to
point toward your concentrator by using its ``set_transform()`` method.

Assemblies do not directly hold surfaces, but only other assemblies or optical
objects.

See also :doc:`assembly`.

Optical Objects
---------------
These are a special kind of assembly, that does not hold other assemblies or
objects, but only surfaces. In the assembly tree, objects hold the leaves.

An object also provides the tracer engine with information on owned rays, and
ray relevancy; owned rays are rays whose next hit will certainly be on a
surface belonging to this object (as in rays refracted into a lens); relevancy
is the ability of determining in advance that some surfaces will not be hit by
a ray. The standard AssembledObject never owns anything and marks all rays as
relevant, but derived classes may change this: ``tracer.models.one_sided_mirror``
for example, marks its sole surface as irrelevant for rays that originate from
it.

See also :doc:`object`

Surfaces
--------
A surface is the basic interacion unit with rays. Each ray is checked for
possible intersections with all surfaces (subject to relevancy information from
the various objects), and the intersection closest to the ray's starting
position is selected for further processing. After a ray is determined to
intersect a surface, that surface is responsible for generating the next
segment of the ray, by reflection or refraction.

Each of the two responsibilities of a surface requires knowlwdge a different
property of the surface: intersection finding is related to the surface's
*Geometry*; generating the resulting rays is related to *optical properties*
such as reflectivity, index of refraction, etc.

This seperation is kept when constructing a surface. Each new Surface instance
requires as arguments a GeometryManager instance, and an optics-callable. The 
GeometryManager class and the optics_callable module documentation details the
required interfaces. The optics callable is a python callable object (e.g. a
function or a class with the ``__call__()`` method) which handles the required
interface.

See also:

* :doc:`surface`
* :doc:`opt_call`
* :doc:`trace_protocol`


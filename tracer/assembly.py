# Defines an assembly class, where an assembly is defined as a collection of AssembledObjects.

import operator
import numpy as N
from spatial_geometry import general_axis_rotation

class Assembly():
    """ Defines an assembly of objects or sub-assemblies.
    Attributes:
    _objects - a list of the objects the assembly contains
    _assemblies - a list of the sub assemblies the assembly contains
    transform - the transformation matrix (written as an array) describing any transformation
    of the assembly into the global coordinates
    """
    def __init__(self, objects=None, subassemblies=None):
        """
        Arguments:
        objects (optional) - a list of AssembledObject instances that are part
            of this assembly.
        subassemblies (optional) - a list of Assembly instances to be
            transformed together with this assembly.
        """
        if objects is None:
            objects = []
        self._objects = objects
        
        if subassemblies is None:
            subassemblies = []
        self._assemblies = subassemblies
        
        self.transform = N.eye(4)
        self.transform_assembly()

    def get_objects(self):
        return self.objects

    def get_assemblies(self):
        return self.assemblies
    
    def get_objects(self):
        """
        Generates a list of AssembledObject instances belonging to this assembly
        or its subassemblies.
        """
        return reduce(operator.add, 
            [asm.get_objects() for asm in self._assemblies] + [self._objects])

    def get_surfaces(self):  
        """
        Generates a list of surface objects out of all the surfaces in the
        objects and subassemblies belonging to this assembly.
        
        The surfaces are guarantied to be in the order that each object returns
        them, and the objects are guarantied to be ordered the same as in 
        self.get_objects()
        """
        return reduce(operator.add, 
            [obj.get_surfaces() for obj in self.get_objects()])

    def add_object(self, object, transform=None):
        """Adds an object to the assembly.
        Arguments: objects - the AssembledObject to add
        transform - the transformation matrix (as an array object) that describes 
        the object in the coordinate system of the Assembly
        """
        if transform == None:
            transform = N.eye(4)
        self._objects.append(object)
        object.set_transform(transform)
        object.transform_object(self.transform)

    def add_assembly(self, assembly, transform=None):
        """Adds an assembly to the current assembly.
        Arguments: assembly - the assembly object to add
        transform - the transformation matrix (as an array object) that describes the 
        new assembly in the coordinate system of the current assembly
        """
        if transform == None:
            transform = N.eye(4)
        self._assemblies.append(assembly)
        assembly.set_transform(transform)
        assembly.transform_assembly(self.transform)

    def set_transform(self, transform):
        self.transform = transform
        self.transform_assembly()

    def get_transform(self):
        return self.transform

    def transform_assembly(self, assembly_transform=N.eye(4)):
        """
        Transforms the entire assembly
        Arguments:
        assembly_transform - the transformation into the parent assembly containing the 
        current assembly
        """
        for object in xrange(len(self._objects)):
            self._objects[object].transform_object(
                N.dot(assembly_transform, self.transform))
        for assembly in xrange(len(self._assemblies)):
            self._assemblies[assembly].transform_assembly(
                N.dot(assembly_transform, self.transform))

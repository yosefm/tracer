# Defines an assembly class, where an assembly is defined as a collection of AssembledObjects.

import numpy as N
from spatial_geometry import general_axis_rotation

class Assembly():
    """ Defines an assembly of objects or sub-assemblies.
    Attributes:
    objects - a list of the objects the assembly contains
    assemblies - a list of the sub assemblies the assembly contains
    transform - the transformation matrix (written as an array) describing any transformation
    of the assembly into the global coordinates
    """
    def __init__(self):
        self.objects = []
        self.assemblies = []
        self.transform = N.eye(4)

    def get_objects(self):
        return self.objects

    def get_assemblies(self):
        return self.assemblies

    def get_surfaces(self):  
        """Gets the total number of surfaces in the entire assembly, for use of calculations
        by the tracer engine
        """
        surfaces = []
        for obj in self.objects:
            surfaces = surfaces + obj.get_surfaces()
        return surfaces 

    def add_object(self, object, transform=None):
        """Adds an object to the assembly.
        Arguments: objects - the AssembledObject to add
        transform - the transformation matrix (as an array object) that describes 
        the object in the coordinate system of the Assembly
        """
        if transform == None:
            transform = N.eye(4)
        self.objects.append(object)
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
        self.assemblies.append(assembly)
        assembly.set_transform(transform)
        assembly.transform_assembly(self.transform)

    def set_transform(self, transform):
        self.transform = transform

    def get_transform(self):
        return self.transform

    def transform_assembly(self, assembly_transform=N.eye(4)):
        """
        Transforms the entire assembly
        Arguments:
        assembly_transform - the transformation into the parent assembly containing the 
        current assembly
        """
        for object in xrange(len(self.objects)):
            self.objects[object].transform_object(N.dot(self.transform,assembly_transform))
        for assembly in xrange(len(self.assemblies)):
            self.assemblies[assembly].transform_assembly(N.dot(self.transform,assembly_transform))
    


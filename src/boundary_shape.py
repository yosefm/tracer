class BoundaryShape():
    

class Plane(BoundaryShape):
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def inters_sphere(self, obj):
        """
        Determines the points of intersection of the plane and a sphere
        """
        # Intersect with a sphere with its center at the origin, then translate
        
        

class Sphere(BoundaryShape):
    def __init__(self, location,  radius):
        self.location = location
        self.radius = radius
    
    def inters_sphere(self, obj):
        """
        Determines the points of intersection of the sphere and another sphere
        """
        
    def inters_parabola(self, obj):
        """
        Determines the points of intersection of the sphere and a parabolic surface
        """
    
    def inters_plane(self, obj):
        """
        Determines the points of intersection of the sphere and a planar surface

        """
        

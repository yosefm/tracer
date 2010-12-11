
from PyQt4 import QtCore, QtGui

class AssemblyTree(QtGui.QTreeWidget):
    """
    Provides a tree-widget for Qt which shows an assembly as a tree of sub-
    assemblies and objects. 
    
    Special tags may be attached to the virgin assembly to make the tree
    provide more information to the user or extend its behaviour. The
    properties are stored in a dictionary in the _tree_tags attribute of a
    Surface, AssembledObject or Assembly instance.
    
    Supported tags:
    caption - string to show instead of default numbering.
    """
    def set_assembly(self, asm):
        self._asm = asm
        self.clear()
        
        # Repopulate:
        asm_item = QtGui.QTreeWidgetItem(["Top assembly"])
        self.addTopLevelItem(asm_item)
        self.expandItem(asm_item)
        self._add_subassembly(asm, asm_item)
        
    def _add_subassembly(self, asm, under):
        """
        Add entries representing a subassembly to the tree, recursively.
        
        Arguments:
        asm - the Assembly object to represent.
        under - the QTreeWidgetItem to place new entries under.
        """
        # Child categories:
        subs = asm.get_assemblies()
        if len(subs) > 0:
            subasms = QtGui.QTreeWidgetItem(["Subassemblies"])
            under.addChild(subasms)
        
        objs = asm.get_local_objects()
        if len(objs) > 0:
            objects = QtGui.QTreeWidgetItem(["Optical objects"])
            under.addChild(objects)
            self._add_objects(asm, objects)
        
        for six in xrange(len(subs)):
            caption = self._get_tag(subs[six], 'caption')
            if caption is None:
                caption = "Assembly %d" % six
            
            asm_item = QtGui.QTreeWidgetItem([caption])
            subasms.addChild(asm_item)
            self._add_subassembly(subs[six], asm_item)
    
    def _add_objects(self, asm, under):
        """
        Add entries representing the objects (leaf nodes) of an assembly to the
        tree.
        """
        objs = asm.get_local_objects()
        for oix in xrange(len(objs)):
            caption = self._get_tag(objs[oix], 'caption')
            if caption is None:
                caption = "Object %d" % oix
            
            tree_obj = QtGui.QTreeWidgetItem([caption])
            under.addChild(tree_obj)
            
            # Add surfaces:
            surfs = objs[oix].get_surfaces()
            for six in xrange(len(surfs)):
                caption = self._get_tag(surfs[six], 'caption')
                if caption is None:
                    caption = "Surface %d" % six
                
                surf_item = QtGui.QTreeWidgetItem([caption])
                tree_obj.addChild(surf_item)
    
    def _get_tag(self, obj, tagname):
        """
        Check if the given object has tags and has the requested tag. If not,
        return None, else return the tag value.
        """
        if not hasattr(obj, '_tree_tags'):
            return None
        return obj._tree_tags.get(tagname, None)

if __name__ == "__main__":
    import sys
    from tracer.models.tau_minidish import standard_minidish
    
    app = QtGui.QApplication(sys.argv)
    ui = AssemblyTree()
    
    asm = standard_minidish(1., 500, 1.)[0]
    # Tags test:
    objs = asm.get_local_objects()
    objs[0]._tree_tags = {'caption': 'Receiver'}
    objs[1]._tree_tags = {'caption': 'Dish'}
    asm.get_assemblies()[0]._tree_tags = {'caption': 'Homogenizer'}
    
    ui.set_assembly(asm)
    ui.show()
    
    sys.exit(app.exec_())

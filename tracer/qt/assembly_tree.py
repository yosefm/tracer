
from PyQt4 import QtCore, QtGui

class AssemblyTree(QtGui.QTreeWidget):
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
            asm_item = QtGui.QTreeWidgetItem(["Assembly %d" % six])
            subasms.addChild(asm_item)
            self._add_subassembly(subs[six], asm_item)
    
    def _add_objects(self, asm, under):
        """
        Add entries representing the objects (leaf nodes) of an assembly to the
        tree.
        """
        objs = asm.get_local_objects()
        for oix in xrange(len(objs)):
            tree_obj = QtGui.QTreeWidgetItem(["Object %d" % oix])
            under.addChild(tree_obj)
            
            # Add surfaces:
            surfs = objs[oix].get_surfaces()
            for six in xrange(len(surfs)):
                surf_item = QtGui.QTreeWidgetItem(["Surface %d" % six])
                tree_obj.addChild(surf_item)


if __name__ == "__main__":
    import sys
    from tracer.models.tau_minidish import standard_minidish
    
    app = QtGui.QApplication(sys.argv)
    ui = AssemblyTree()
    
    asm = standard_minidish(1., 500, 1.)[0]
    ui.set_assembly(asm)
    ui.show()
    
    sys.exit(app.exec_())

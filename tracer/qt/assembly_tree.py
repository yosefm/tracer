
from PyQt4 import QtCore, QtGui

class AssemblyTreeItem(QtGui.QTreeWidgetItem):
    """
    A QTreeWidgetItem that keeps track of the corresponding assembly item and
    Updates it as necessary. The assembly item ,ay be instance of Assembly,
    AssembledObject or Surface.
    """
    def __init__(self, captions, asm_item):
        QtGui.QTreeWidgetItem.__init__(self, captions)
        self._asm_item = asm_item
        
        # Since the item is graphically managed, it gets the data structure
        # that the tree expects, holding the data specific to this widget.
        if not hasattr(asm_item, '_tree_tags'):
            asm_item._tree_tags = {}
    
    def update_caption(self, updated):
        """
        Copy the caption from the graphical item to the model.
        
        Arguments:
        updated - the updated item object. If it isn't self, nothing will be
            done.
        """
        if updated is not self:
            return
        
        self._asm_item._tree_tags['caption'] = self.text(0)
        
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
        asm_item = AssemblyTreeItem(["Top assembly"], asm)
        asm_item.setFlags(asm_item.flags()|QtCore.Qt.ItemIsEditable)
        self.itemChanged.connect(asm_item.update_caption)
        
        self.addTopLevelItem(asm_item)
        self.expandItem(asm_item)
        self._add_subassembly(asm, asm_item)
    
    def get_assembly(self):
        return self._asm
        
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
            caption = self.get_tag(subs[six], 'caption')
            if caption is None:
                caption = "Assembly %d" % six
                self.set_tag(subs[six], 'caption', caption)
            
            asm_item = AssemblyTreeItem([caption], subs[six])
            asm_item.setFlags(asm_item.flags()|QtCore.Qt.ItemIsEditable)
            self.itemChanged.connect(asm_item.update_caption)
            
            subasms.addChild(asm_item)
            self._add_subassembly(subs[six], asm_item)
    
    def _add_objects(self, asm, under):
        """
        Add entries representing the objects (leaf nodes) of an assembly to the
        tree.
        """
        objs = asm.get_local_objects()
        for oix in xrange(len(objs)):
            caption = self.get_tag(objs[oix], 'caption')
            if caption is None:
                caption = "Object %d" % oix
                self.set_tag(objs[oix], 'caption', caption)
            
            tree_obj = AssemblyTreeItem([caption], objs[oix])
            tree_obj.setFlags(tree_obj.flags()|QtCore.Qt.ItemIsEditable)
            self.itemChanged.connect(tree_obj.update_caption)

            under.addChild(tree_obj)
            
            # Add surfaces:
            surfs = objs[oix].get_surfaces()
            for six in xrange(len(surfs)):
                caption = self.get_tag(surfs[six], 'caption')
                if caption is None:
                    caption = "Surface %d" % six
                    self.set_tag(surfs[six], 'caption', caption)
                
                surf_item = AssemblyTreeItem([caption], surfs[six])
                surf_item.setFlags(surf_item.flags()|QtCore.Qt.ItemIsEditable)
                self.itemChanged.connect(surf_item.update_caption)
                
                tree_obj.addChild(surf_item)
    
    def get_tag(self, obj, tagname):
        """
        Check if the given object has tags and has the requested tag. If not,
        return None, else return the tag value.
        
        Arguments:
        obj - an assembly object (Assembly, AssembledObject or Surface)
        tagname - name of tag to to fetch.
        """
        if not hasattr(obj, '_tree_tags'):
            return None
        return obj._tree_tags.get(tagname, None)
    
    def set_tag(self, obj, tagname, value):
        """
        Create or change a tag on an assembly member.
        
        Arguments:
        obj - an assembly object (Assembly, AssembledObject or Surface)
        tagname - name of tag to set or create
        value - the value to set tagname to. A string.
        """
        if not hasattr(obj, '_tree_tags'):
            obj._tree_tags = {}
        
        obj._tree_tags[tagname] = value

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

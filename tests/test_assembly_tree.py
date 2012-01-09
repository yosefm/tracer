# Test that the AssemblyTree object changes properties as it should

import unittest, sys
from PyQt4.QtGui import QApplication
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt, QPoint, QRect

from tracer.qt.assembly_tree import AssemblyTree
from tracer.models.tau_minidish import standard_minidish

class TestAssemblyTreeWidget(unittest.TestCase):
    def setUp(self):
        self.app = QApplication(sys.argv)
        self.form = AssemblyTree()
        
        asm = standard_minidish(1., 500, 1.)[0]
        self.form.set_assembly(asm)
    
    def test_numbering(self):
        """Default naming of tree items"""
        asm = self.form.get_assembly()
        objs = asm.get_local_objects()
        homog = asm.get_assemblies()[0]
        
        self.assertEqual(self.form.get_tag(objs[0], 'caption'), 'Object 0')
        self.assertEqual(self.form.get_tag(objs[1], 'caption'), 'Object 1')
        self.assertEqual(self.form.get_tag(homog, 'caption'), 'Assembly 0')
    
    def test_renaming(self):
        """Renaming tree items graphically"""
        objs = self.form.get_assembly().get_local_objects()
        obj_items = self.form.topLevelItem(0).child(1)
        
        dish_pos = self.form.visualItemRect(obj_items.child(0))
        obj_items.child(0).setText(0, "edited")
        self.assertEqual(self.form.get_tag(objs[0], 'caption'), 'edited')
    
    def tearDown(self):
        del self.app

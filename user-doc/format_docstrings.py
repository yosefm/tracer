# A Sphinx extension that, in collaboration with the autodoc extension,
# parses the docstring format used in this project.

import re

# Section processing regexes:
end_section = re.compile(r'^\s*$')
section_title = re.compile(r'^\s*(\w+):')
def_line = re.compile(r'(\w+)\s*-\s*(.+)')

def setup(app):
    app.connect('autodoc-process-docstring', parse_docstring)

def parse_docstring(app, what, name, obj, options, lines):
    """
    Divides the docstring into sections, separated by an empyt line in the
    docstring. If a section starts with a line like "$some_title:", then it is
    turned into a titled section with $some_title as its title. 
    
    A line starting with a python identifier, then a dash, then a definition,
    is turned into a definition-list item.
    """
    out_lines = []
    place = 0
    section = []
    bullet = False
    for place in xrange(len(lines)):
        line = lines[place]
        
        # End section: flush into out_lines.
        m = end_section.match(line)
        if m is not None:
            out_lines.extend(section + [line])
            section = []
            bullet = False
            continue
        
        if len(section) == 0:
            # Start of a title-section:
            m = section_title.match(line)
            if m is not None:
                section.extend(['*' + m.group(1) + '*', ''])
                continue
        
        m = def_line.match(line)
        if m is not None:
            out_lines.extend(section + [''])
            section = []
            bullet = True
            section.extend(['', '* ' + m.group(1) + ': ' + m.group(2)])
            continue
        
        line = line.strip()
        if bullet:
            line = '  ' + line
        section.append(line)
    
    # Flush last section:
    out_lines.extend(section)
    section=[]
    
    # In-place modification of `lines` is needed:
    lines[:] = out_lines


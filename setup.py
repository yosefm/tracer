#!/usr/bin/env python

from distutils.core import setup

setup(name='Tracer',
      version='0.2',
      description='Ray-tracing library in Python, focused on solar energy research',
      author='The Tracer developers',
      author_email='tracer-user@lists.berlios.de',
      url='http://tracer.berlios.de',
      packages=['tracer', 'tracer.models', 'tracer.mayavi_ui'],
      license="GPL v3.0 or later, see LICENSE file"
     )


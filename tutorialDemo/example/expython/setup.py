#! /usr/bin/env python

from distutils.core import setup, Extension

MOD = "Itcastcpp"

setup(name=MOD, ext_modules=[Extension(MOD, sources=['Itcastcpp.c', 'Itcastcppwrapper.c'])])

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fmdtools CLI package for scaffolding and template generation.
"""

from .main import app, main
from .schemas import LevelSpec, FunctionSpec, FlowSpec, ArchitectureSpec
from .generate import render_level
from .ai_adapter import AIWizard

__all__ = [
    'app',
    'main', 
    'LevelSpec',
    'FunctionSpec',
    'FlowSpec',
    'ArchitectureSpec',
    'render_level',
    'AIWizard'
]


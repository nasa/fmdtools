#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compatibility layer for fmdtools imports.

This module provides a stable import interface that handles
different fmdtools package structures and versions.
"""

try:
    from fmdtools.define.block.function import Function
except ImportError:
    try:
        from fmdtools.define.function import Function
    except ImportError:
        raise ImportError("Could not import Function from fmdtools")

try:
    from fmdtools.define.block.base import Block
except ImportError:
    try:
        from fmdtools.define.base import Block
    except ImportError:
        raise ImportError("Could not import Block from fmdtools")

try:
    from fmdtools.define.container.state import State
except ImportError:
    try:
        from fmdtools.define.state import State
    except ImportError:
        raise ImportError("Could not import State from fmdtools")

try:
    from fmdtools.define.container.mode import Mode
except ImportError:
    try:
        from fmdtools.define.mode import Mode
    except ImportError:
        raise ImportError("Could not import Mode from fmdtools")

try:
    from fmdtools.define.container.parameter import Parameter
except ImportError:
    try:
        from fmdtools.define.parameter import Parameter
    except ImportError:
        raise ImportError("Could not import Parameter from fmdtools")

try:
    from fmdtools.define.flow.base import Flow
except ImportError:
    try:
        from fmdtools.define.flow import Flow
    except ImportError:
        raise ImportError("Could not import Flow from fmdtools")

try:
    from fmdtools.define.architecture.function import FunctionArchitecture, FunctionArchitectureGraph
except ImportError:
    try:
        from fmdtools.define.architecture import FunctionArchitecture, FunctionArchitectureGraph
    except ImportError:
        raise ImportError("Could not import FunctionArchitecture from fmdtools")

# Test that we can import propagate (optional but helpful)
try:
    from fmdtools.sim import propagate as _prop
except ImportError:
    pass  # Optional, but helps catch major import issues

__all__ = [
    'Function',
    'Block', 
    'State',
    'Mode',
    'Parameter',
    'Flow',
    'FunctionArchitecture',
    'FunctionArchitectureGraph'
]


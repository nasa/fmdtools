#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for the fmdtools CLI.
"""

import re
from typing import Any, Dict, Tuple


def slugify_module(s: str) -> str:
    """Convert string to valid Python module name."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s).strip("_")
    if re.match(r"^[0-9]", s):
        s = "_" + s
    return s.lower()


def to_class_name(s: str) -> str:
    """Convert string to valid Python class name."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s).title().replace(" ", "")
    if re.match(r"^[0-9]", s):
        s = "_" + s
    return s


def sanitize_names(spec_name: str, function_names: list, flow_names: list) -> Tuple[str, Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
    """Sanitize all names and return mappings."""
    safe_spec_name = slugify_module(spec_name)
    
    # Sanitize function names
    function_mapping = {}
    safe_function_names = []
    for name in function_names:
        safe_name = slugify_module(name)
        class_name = to_class_name(name)
        function_mapping[name] = (safe_name, class_name)
        safe_function_names.append(safe_name)
    
    # Sanitize flow names
    flow_mapping = {}
    safe_flow_names = []
    for name in flow_names:
        safe_name = slugify_module(name)
        class_name = to_class_name(name)
        flow_mapping[name] = (safe_name, class_name)
        safe_flow_names.append(safe_name)
    
    return safe_spec_name, function_mapping, flow_mapping


def is_number(value: Any) -> bool:
    """Check if value is a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


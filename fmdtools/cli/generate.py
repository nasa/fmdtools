#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code generator for fmdtools models from specifications.

This module takes LevelSpec objects and renders them into
Python files using Jinja2 templates.
"""

import os
import platform
from pathlib import Path
from typing import List

import typer
from jinja2 import Environment, PackageLoader, select_autoescape

from .schemas import LevelSpec, FunctionSpec, FlowSpec, ArchitectureSpec
from .utils import sanitize_names


def render_level(spec: LevelSpec, out_dir: str = ".", force: bool = False, dry_run: bool = False) -> List[Path]:
    """Render a complete level model from specification."""
    # Sanitize names
    safe_spec_name, function_mapping, flow_mapping = sanitize_names(
        spec.name, 
        [f.name for f in spec.functions], 
        [f.name for f in spec.flows]
    )
    
    out_path = Path(out_dir) / safe_spec_name
    
    if not dry_run:
        out_path.mkdir(parents=True, exist_ok=True)
    
    # Log version info and output path
    if not dry_run:
        try:
            import fmdtools  # type: ignore
            typer.echo(f"fmdtools {getattr(fmdtools, '__version__', 'unknown')} • Python {platform.python_version()}")
        except Exception:
            typer.echo(f"fmdtools (unknown version) • Python {platform.python_version()}")
        typer.echo(f"Output: {Path(out_dir).resolve() / safe_spec_name}")
    
    files = []
    
    # Check for existing files if not forcing (case-insensitive on Windows)
    if not force and not dry_run:
        existing_files = []
        for func in spec.functions:
            safe_name = function_mapping[func.name][0]
            if (out_path / f"{safe_name}.py").exists():
                existing_files.append(f"{safe_name}.py")
        if (out_path / "flows.py").exists():
            existing_files.append("flows.py")
        if (out_path / "architecture.py").exists():
            existing_files.append("architecture.py")
        if (out_path / f"level_{safe_spec_name}.py").exists():
            existing_files.append(f"level_{safe_spec_name}.py")
        
        if existing_files:
            raise FileExistsError(
                f"Files already exist in {out_path}: {', '.join(existing_files)}. "
                f"Use --force to overwrite or --dry-run to preview."
            )
    
    # Generate function files
    for func in spec.functions:
        safe_name, class_name = function_mapping[func.name]
        target = out_path / f"{safe_name}.py"
        content = render_function(func, spec, class_name)
        if dry_run:
            print(f"\n--- {target} ---")
            print(content)
        else:
            target.write_text(content)
        files.append(target)
    
    # Generate flows file
    if spec.flows:
        target = out_path / "flows.py"
        content = render_flows(spec, flow_mapping)
        if dry_run:
            print(f"\n--- {target} ---")
            print(content)
        else:
            target.write_text(content)
        files.append(target)
    
    # Generate architecture file
    target = out_path / "architecture.py"
    content = render_architecture(spec, function_mapping, flow_mapping)
    if dry_run:
        print(f"\n--- {target} ---")
        print(content)
    else:
        target.write_text(content)
    files.append(target)
    
    # Generate main level file
    target = out_path / f"level_{safe_spec_name}.py"
    content = render_main_level(spec, safe_spec_name)
    if dry_run:
        print(f"\n--- {target} ---")
        print(content)
    else:
        target.write_text(content)
    files.append(target)
    
    # Generate __init__.py
    target = out_path / "__init__.py"
    content = render_init(spec, safe_spec_name)
    if dry_run:
        print(f"\n--- {target} ---")
        print(content)
    else:
        target.write_text(content)
    files.append(target)
    
    # Generate README.md
    target = out_path / "README.md"
    content = render_readme(spec, safe_spec_name, function_mapping, flow_mapping)
    if dry_run:
        print(f"\n--- {target} ---")
        print(content)
    else:
        target.write_text(content)
    files.append(target)
    
    return files


def render_function(func: FunctionSpec, spec: LevelSpec, class_name: str) -> str:
    """Render a function template."""
    template = env.get_template("function.py.j2")
    return template.render(func=func, spec=spec, class_name=class_name)


def render_flows(spec: LevelSpec, flow_mapping: dict) -> str:
    """Render flows template."""
    template = env.get_template("flows.py.j2")
    return template.render(spec=spec, flows=spec.flows, flow_mapping=flow_mapping)


def render_architecture(spec: LevelSpec, function_mapping: dict, flow_mapping: dict) -> str:
    """Render architecture template."""
    template = env.get_template("architecture.py.j2")
    # Create a proper architecture class name
    arch_class_name = spec.architecture.name.replace(' ', '').replace('!', '').replace('@', '').replace('#', '').replace('$', '').replace('%', '').replace('^', '').replace('&', '').replace('*', '').replace('(', '').replace(')', '').replace('-', '').replace('+', '').replace('=', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '').replace('|', '').replace(';', '').replace(':', '').replace('"', '').replace("'", '').replace(',', '').replace('.', '').replace('<', '').replace('>', '').replace('/', '').replace('?', '')
    if arch_class_name[0].isdigit():
        arch_class_name = '_' + arch_class_name
    return template.render(spec=spec, arch=spec.architecture, functions=spec.functions, 
                         function_mapping=function_mapping, flow_mapping=flow_mapping,
                         arch_class_name=arch_class_name)


def render_main_level(spec: LevelSpec, safe_spec_name: str) -> str:
    """Render main level template."""
    template = env.get_template("level.py.j2")
    # Create a proper class name from the safe spec name
    class_name = safe_spec_name.replace('_', '').replace('-', '').title()
    if class_name[0].isdigit():
        class_name = '_' + class_name
    # Create a proper architecture class name
    arch_class_name = spec.architecture.name.replace(' ', '').replace('!', '').replace('@', '').replace('#', '').replace('$', '').replace('%', '').replace('^', '').replace('&', '').replace('*', '').replace('(', '').replace(')', '').replace('-', '').replace('+', '').replace('=', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('\\', '').replace('|', '').replace(';', '').replace(':', '').replace('"', '').replace("'", '').replace(',', '').replace('.', '').replace('<', '').replace('>', '').replace('/', '').replace('?', '')
    if arch_class_name[0].isdigit():
        arch_class_name = '_' + arch_class_name
    return template.render(spec=spec, safe_spec_name=safe_spec_name, class_name=class_name, arch_class_name=arch_class_name)


def render_init(spec: LevelSpec, safe_spec_name: str) -> str:
    """Render __init__.py template."""
    template = env.get_template("init.py.j2")
    return template.render(spec=spec, safe_spec_name=safe_spec_name)


def render_readme(spec: LevelSpec, safe_spec_name: str, function_mapping: dict, flow_mapping: dict) -> str:
    """Render README.md template."""
    template = env.get_template("README.md.j2")
    return template.render(
        spec=spec,
        safe_spec_name=safe_spec_name,
        function_mapping=function_mapping,
        flow_mapping=flow_mapping
    )


# Set up Jinja2 environment with custom filters and better defaults
env = Environment(
    loader=PackageLoader("fmdtools.cli", "templates"),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True
)
env.filters["pyrepr"] = repr

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive question system for building fmdtools specifications.

This module handles the interactive prompts to gather information
from users about their desired model structure.
"""

import ast
from typing import List, Optional
import typer

from .schemas import (
    LevelSpec, FunctionSpec, FlowSpec, ArchitectureSpec, 
    ConnectionSpec, Fault, SimulationSpec
)
from .utils import slugify_module, to_class_name


def _cast(s: str):
    """Safely convert string input to Python literal."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def ask_level_spec(default_name: Optional[str] = None, no_input: bool = False) -> LevelSpec:
    """Interactively gather specification for a level model."""
    if not no_input:
        typer.echo("\nfmdtools Model Builder")
        typer.echo("=" * 40)
    
    # Get basic model information
    name = default_name or (typer.prompt("Model name", default="MyModel") if not no_input else "MyModel")
    
    description = typer.prompt("Model description (optional)", default="") if not no_input else ""
    
    # Get functions
    if not no_input:
        typer.echo(f"\nFunctions for {name}:")
    functions = []
    
    # Ensure at least one function
    if no_input:
        functions.append(FunctionSpec(
            name=f"{name}Function",
            states={"value": 1.0},
            modes=["nominal"],
            faults=[]
        ))
    else:
        while True:
            add_function = typer.confirm("Add a function?", default=True)
            if not add_function:
                break
            
            func_name = typer.prompt("Function name")
            if not func_name.strip():
                typer.echo("Function name cannot be empty")
                continue
            func_desc = typer.prompt("Function description (optional)", default="")
            
            # Get states - ensure at least one state variable
            states = {}
            if typer.confirm("Add state variables?", default=True):
                while True:
                    state_name = typer.prompt("State variable name")
                    if not state_name:
                        break
                    if not state_name.strip():
                        typer.echo("State variable name cannot be empty")
                        continue
                    state_value = _cast(typer.prompt("Default value", default="1.0"))
                    states[state_name] = state_value
                    if not typer.confirm("Add another state variable?", default=False):
                        break
            
            # Get modes - ensure at least nominal mode
            modes = ["nominal"]
            if typer.confirm("Add additional modes?", default=False):
                while True:
                    mode_name = typer.prompt("Mode name")
                    if not mode_name:
                        break
                    if not mode_name.strip():
                        typer.echo("Mode name cannot be empty")
                        continue
                    modes.append(mode_name)
                    if not typer.confirm("Add another mode?", default=False):
                        break
            
            # Get faults
            faults = []
            if typer.confirm("Add fault modes?", default=False):
                while True:
                    add_fault = typer.confirm("Add another fault?", default=False)
                    if not add_fault:
                        break
                    
                    fault_name = typer.prompt("Fault name")
                    if not fault_name.strip():
                        typer.echo("Fault name cannot be empty")
                        continue
                    fault_detectable = typer.confirm("Is this fault detectable?", default=False)
                    faults.append(Fault(
                        name=fault_name,
                        detection=fault_detectable
                    ))
            
            functions.append(FunctionSpec(
                name=func_name,
                description=func_desc,
                states=states,
                modes=modes,
                faults=faults
            ))
    
    # Get flows
    if not no_input:
        typer.echo(f"\nFlows for {name}:")
    flows = []
    
    # Ensure at least one flow
    if no_input:
        flows.append(FlowSpec(
            name=f"{name}Flow",
            vars={"rate": 1.0},
            description=""
        ))
    else:
        while True:
            add_flow = typer.confirm("Add a flow?", default=True)
            if not add_flow:
                break
            
            flow_name = typer.prompt("Flow name")
            if not flow_name.strip():
                typer.echo("Flow name cannot be empty")
                continue
            flow_desc = typer.prompt("Flow description (optional)", default="")
            
            # Get flow variables - ensure at least one variable
            vars = {}
            if typer.confirm("Add flow variables?", default=True):
                while True:
                    var_name = typer.prompt("Variable name")
                    if not var_name:
                        break
                    if not var_name.strip():
                        typer.echo("Variable name cannot be empty")
                        continue
                    var_value = _cast(typer.prompt("Default value", default="1.0"))
                    vars[var_name] = var_value
                    if not typer.confirm("Add another variable?", default=False):
                        break
            
            flows.append(FlowSpec(
                name=flow_name,
                description=flow_desc,
                vars=vars
            ))
    
    # Get architecture
    if not no_input:
        typer.echo(f"\nArchitecture for {name}:")
    arch_name = f"{name}Arch"
    
    # Get connections
    connections = []
    if len(functions) > 1 and not no_input:
        typer.echo("\nFunction connections:")
        for i, func in enumerate(functions):
            for j, other_func in enumerate(functions):
                if i != j:
                    connect = typer.confirm(
                        f"Connect {func.name} -> {other_func.name}?",
                        default=False
                    )
                    if connect:
                        flow_name = typer.prompt(
                            f"Flow name for {func.name} -> {other_func.name}",
                            default=f"{func.name}To{other_func.name}"
                        )
                        if not flow_name.strip():
                            typer.echo("Flow name cannot be empty")
                            continue
                        connections.append(ConnectionSpec(
                            from_fn=func.name,
                            to_fn=other_func.name,
                            flow_name=flow_name
                        ))
    
    # Get simulation preferences
    if not no_input:
        typer.echo(f"\nSimulation options for {name}:")
    sample_run = True if no_input else typer.confirm("Include basic sample run?", default=True)
    fault_analysis = False if no_input else typer.confirm("Include fault analysis?", default=False)
    parameter_study = False if no_input else typer.confirm("Include parameter study?", default=False)
    
    simulation = SimulationSpec(
        sample_run=sample_run,
        fault_analysis=fault_analysis,
        parameter_study=parameter_study
    )
    
    # Create architecture spec
    architecture = ArchitectureSpec(
        name=arch_name,
        functions=[f.name for f in functions],
        connections=connections
    )
    
    # Create and return the complete specification
    return LevelSpec(
        name=name,
        description=description,
        functions=functions,
        flows=flows,
        architecture=architecture,
        simulation=simulation
    )


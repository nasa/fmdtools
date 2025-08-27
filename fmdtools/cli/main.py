#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fmdtools CLI - Command line interface for scaffolding and template generation.

This CLI provides tools to create fmdtools models from specifications,
either through interactive prompts, programmatic configuration, or AI wizard.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from typer import Exit

from .schemas import LevelSpec, FunctionSpec, FlowSpec, ArchitectureSpec
from .generate import render_level
from .questions import ask_level_spec
from .utils import sanitize_names
from .ai_adapter import AIWizard

app = typer.Typer(
    help="fmdtools scaffolding CLI - Create models from specifications",
    add_completion=False
)


@app.command("create")
def create_level(
    kind: str = typer.Argument(..., help="Type of template to create (level|function)"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for the model"),
    out: str = typer.Option(".", "--out", "-o", help="Output directory"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Minimal prompts, use defaults"),
    ai: bool = typer.Option(False, "--ai", help="Enable AI wizard mode"),
    model: Optional[str] = typer.Option(None, "--model", help="Override AI model (default: from .env)"),
    desc: Optional[str] = typer.Option(None, "--desc", help="Seed description for AI wizard"),
    confirm: bool = typer.Option(False, "--confirm", help="Ask for confirmation before generating files"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print outputs; write nothing"),
    print_spec: bool = typer.Option(False, "--print-spec", help="Print the specification JSON"),
    write_spec: Optional[str] = typer.Option(None, "--write-spec", help="Write specification to JSON file"),
    no_input: bool = typer.Option(False, "--no-input", help="Disable prompts and confirmations"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress specification output"),
):
    """Create a new fmdtools level model."""
    if kind != "level":
        typer.echo("Only 'level' scaffolding is implemented currently", err=True)
        raise Exit(2)
    
    # Handle no-input mode
    if no_input:
        confirm = False
    
    # Set confirm default based on quick mode
    if quick and not no_input:
        confirm = True
    
    # AI Wizard Mode
    if ai:
        try:
            typer.echo("Starting AI Wizard...")
            wiz = AIWizard(model=model)
            
            # Add description seed if provided
            if desc:
                wiz.messages.append({"role": "user", "content": f"System description: {desc}"})
            
            spec = wiz.run()
            
            # If user invoked quick, keep the flag so post-init can auto-add a stub flow when needed
            spec.is_quick_mode = spec.is_quick_mode or quick
            
        except Exception as e:
            typer.echo(f"AI wizard failed: {e}", err=True)
            typer.echo("\nTips:")
            typer.echo("  • Ensure you have a .env file with OPENAI_API_KEY")
            typer.echo("  • Check your OpenAI API quota and rate limits")
            typer.echo("  • Try interactive mode instead: fmdtools create level")
            raise Exit(1)
    
    else:
        # Standard interactive/quick mode
        if quick and name:
            # Quick mode with minimal configuration
            spec = LevelSpec(
                name=name,
                functions=[
                    FunctionSpec(name=f"{name}Function", states={"value": 1.0})
                ],
                flows=[
                    FlowSpec(name=f"{name}Flow", vars={"rate": 1.0})
                ],
                architecture=ArchitectureSpec(
                    name=f"{name}Arch",
                    functions=[f"{name}Function"],
                    connections=[]
                ),
                is_quick_mode=True
            )
        else:
            # Interactive mode
            spec = ask_level_spec(default_name=name, no_input=no_input)
            spec.is_quick_mode = quick
    
    # Auto-create stub flow if needed and in quick mode
    if not spec.flows and spec.is_quick_mode:
        spec.flows.append(FlowSpec(name=f"{spec.name}Flow", vars={"rate": 1.0}))
    
    # Validate specification
    try:
        spec.model_post_init(None)
    except ValueError as e:
        typer.echo(f"Specification validation failed: {e}", err=True)
        raise Exit(1)
    
    # Write specification to file if requested
    if write_spec:
        try:
            with open(write_spec, 'w') as f:
                json.dump(spec.model_dump(), f, indent=2)
            typer.echo(f"Specification written to: {write_spec}")
        except Exception as e:
            typer.echo(f"Failed to write specification: {e}", err=True)
            raise Exit(1)
    
    # Show planned specification (unless quiet)
    if not quiet:
        if print_spec:
            typer.echo("\nSpecification JSON:")
            typer.echo(spec.model_dump_json(indent=2))
        
        typer.echo("\nPlanned specification:")
        typer.echo(f"Model: {spec.name}")
        typer.echo(f"Functions: {len(spec.functions)}")
        typer.echo(f"Flows: {len(spec.flows)}")
        typer.echo(f"Connections: {len(spec.architecture.connections)}")
    
    # Confirm before generating files
    if confirm and not no_input and not typer.confirm("\nGenerate files?", default=True):
        typer.echo("Operation cancelled.")
        raise Exit()
    
    # Generate files
    try:
        paths = render_level(spec, out_dir=out, force=force, dry_run=dry_run)
        if dry_run:
            typer.echo("\nDry run completed. No files written.")
        else:
            typer.echo(f"\nSuccessfully created {len(paths)} files:")
            for path in paths:
                typer.echo(f"  {path}")
            # Show absolute output path
            output_path = Path(out).resolve() / spec.name.lower()
            typer.echo(f"\nWrote {len(paths)} files to {output_path}")
    except Exception as e:
        typer.echo(f"\nError generating files: {e}", err=True)
        raise Exit(1)


@app.command("validate")
def validate_spec(
    spec_file: str = typer.Argument(..., help="Path to specification JSON file"),
):
    """Validate a specification file."""
    try:
        with open(spec_file, 'r') as f:
            spec_data = json.load(f)
        
        spec = LevelSpec.model_validate(spec_data)
        spec.model_post_init(None)  # Run validation
        typer.echo("Specification is valid!")
        typer.echo(f"Model: {spec.name}")
        typer.echo(f"Functions: {len(spec.functions)}")
        typer.echo(f"Flows: {len(spec.flows)}")
        typer.echo(f"Connections: {len(spec.architecture.connections)}")
        
    except Exception as e:
        typer.echo(f"Invalid specification: {e}", err=True)
        raise Exit(1)


@app.command("lint")
def lint_level(
    path: str = typer.Argument(..., help="Path to generated level directory"),
):
    """Lint a generated level by importing and running a basic simulation."""
    try:
        import importlib.util
        import sys
        from pathlib import Path
        
        level_path = Path(path)
        if not level_path.exists():
            typer.echo(f"Path does not exist: {path}", err=True)
            raise Exit(1)
        
        # Find the main level file
        level_files = list(level_path.glob("level_*.py"))
        if not level_files:
            typer.echo(f"No level_*.py file found in {path}", err=True)
            raise Exit(1)
        
        level_file = level_files[0]
        
        # Import the module
        modspec = importlib.util.spec_from_file_location("level_module", level_file)
        module = importlib.util.module_from_spec(modspec)
        modspec.loader.exec_module(module)
        
        # Get the MODEL_CLASS constant
        model_class = getattr(module, "MODEL_CLASS", None)
        if not model_class:
            typer.echo("MODEL_CLASS not exported from level file", err=True)
            raise Exit(1)
        
        # Try to run a basic simulation
        try:
            import fmdtools.sim.propagate as prop
            
            model = model_class(track='all')
            
            # Handle different propagate.nominal signatures
            try:
                result, mdlhist = prop.nominal(model, end_time=1)
                # Check if result has time attribute
                if hasattr(result, 'time'):
                    final_time = result.time
                elif isinstance(result, dict) and 'time' in result:
                    final_time = result['time']
                else:
                    final_time = 'unknown'
            except TypeError:
                # Old signature
                mdlhist = prop.nominal(model, end_time=1)
                final_time = 1
            
            typer.echo("Lint passed! Model imported and ran successfully.")
            typer.echo(f"Final time: {final_time}")
            
        except Exception as e:
            typer.echo(f"Lint failed during simulation: {e}", err=True)
            raise Exit(1)
            
    except Exception as e:
        typer.echo(f"Lint failed: {e}", err=True)
        raise Exit(1)


@app.command("list-templates")
def list_templates():
    """List available template types."""
    typer.echo("Available template types:")
    typer.echo("  level - Complete system model with functions, flows, and architecture")
    typer.echo("  function - Individual function block (coming soon)")
    typer.echo("  flow - Flow definition (coming soon)")


@app.command("ai-setup")
def ai_setup():
    """Help users set up AI wizard mode."""
    typer.echo("AI Wizard Setup")
    typer.echo("=" * 30)
    typer.echo("To use the AI wizard, you need:")
    typer.echo("1. An OpenAI API key")
    typer.echo("2. A .env file in the fmdtools directory")
    typer.echo("\nCreate a .env file with:")
    typer.echo("OPENAI_API_KEY=sk-your-api-key-here")
    typer.echo("FMDTOOLS_AI_MODEL=gpt-4o-mini")
    typer.echo("\nThen run:")
    typer.echo("fmdtools create level --ai")
    typer.echo("\nThe AI will guide you through creating your model!")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

# fmdtools Python Library

## Project Overview

See: @README.md and @docs-source/Intro_to_fmdtools.md

## Best Practices

See: @docs-source/Development Guide.rst

## Library Usage

### General

First see provided docstrings in `src/fmdtools` for minimal example usages.

Then refer to `examples` for in-depth examples.

### Model Development

Develop models using the `define` package located at `src/fmdtools/define`.

### Simulation

Simulate models using the `sim` package located at `src/fmdtools/sim`.

- Use `propagate` and its contained methods
- Use `sample` to define custom scenario samples
- Use `scenario` to define custom scenarios (if not already covered by `propagate` defaults)
- Use `search` to optimize scenarios or model parameters over simulation outcomes

### Analysis

Provide model analysis outputs (plots, tables metrics, etc.) using functionality provided by the `analyze` package at `src/fmdtools/analyze`

- The `history` module defines model histories. Use these methods for plotting and analysis of simulation histories.
- The `result` module defines model results. Use these methods for plotting, metrics quantification, and analysis of simulation results.
- The `phases` module is used to determine phases of operation from a history for better fault sampling.
- The `tabulate` module is used to provide tables of statistical metrics of interest as well as FMEA-style analyses.
- The `graph` sub-package is used to display the model architecture, including interactions and containment structure

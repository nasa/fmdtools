# fmdtools Python Library

## Project Overview

See: @README.md and @docs-source/Intro_to_fmdtools.md

## Best Practices

See: @docs-source/best-practices.md

## Library Usage

### General

- Refer to code directly in `src/fmdtools/` before going off of the examples provided in `examples`.

### Model Development

- Develop models using the `define` package located at `src/fmdtools/define`.
- The main modeling concepts are Containers, Flows, Blocks, and Architectures, where:
    - Architectures are used to connect Blocks via Flows
    - Blocks define behavior
    - Flows represent shared variables
    - Containers are used to store variables within Blocks or Flows.

#### Structure

First, determine the overall high-level structure of the model. It may be appropriate to ask the user about this.
- If the model is going to have a number of interacting behaviors, use an architecture:
	- Most fmdtools models a `FunctionArchitecture` classes that are built out of `Function` and `Flow` classes
	- If the model is to be a discrete set of actions, build an `ActionArchitecture` class built out of `Action` and `Flow` classes instead
	- If the model is a set of components, a `ComponentArchitecture` class may be used, but check with the user since they may prefer a `FunctionArchitecture`
- If the model is going to have a single behavior, use a block:
	- By default, use a `Function` for most single-block models
	- If the model is of a discrete action, use an `Action`
- Otherwise, it may be possible to use other modeling constructs directly depending on the use-case. Stop and sk the user if this is their preference.


Second, apply the relevant skills for each model element:

- For architectures, use the [Architecture Development Skill](.agents/skills/arch-dev/SKILL.md)

- For blocks, use the [Block Development Skill](.agents/skills/block-dev/SKILL.md)

- For containers, refer to the relevant docs in `src/fmdtools/define/container` as well as the [Container Code Templates](docs-source/Intro_to_fmdtools.md#containers---the-building-blocks-of-simulations-smaller)

Third, determine the correct file structure for to write the model to:

- For a small model (<1000 lines), a monolythic file is fine

- Otherwise, split the file up such that more complicated blocks are given their own files.

- Each file should have the following structure:
	- imports
	- Shared Containers
	- Flow Containers followed by their Flows
	- Block Containers followed by their Blocks
	- Architecture Containers followed by the Architecture
	- Short script initializing and verifying behavior from the various blocks and architectures using "if __name__ == "__main__":" protection statement.

- Use the naming conventions for files specified in @docs-source/best_practices.md#Structuring-your-Project-Repository

### Simulation

Simulate models using the `sim` package located at `src/fmdtools/sim`.

- Use `propagate` and its contained methods
- Use `sample` to define custom scenario samples
- Use `scenario` to define custom scenarios (if not already covered by `propagate` defaults)
- Use `search` to optimize scenarios or model parameters over simulation outcomes

### Analysis

Provide model analysis outputs (plots, tables metrics, etc.) using functionality provided by the `analyze` package at `src/fmdtools/analyze`:
    - The `history` module defines model histories. Use these methods for plotting and analysis of simulation histories.
    - The `result` module defines model results. Use these methods for plotting, metrics quantification, and analysis of simulation results.
    - The `phases` module is used to determine phases of operation from a history for better fault sampling.
    - The `tabulate` module is used to provide tables of statistical metrics of interest as well as FMEA-style analyses.
    - The `graph` sub-package is used to display the model architecture, including interactions and containment structure

- Use built-in methods to Result and History (the outputs of methods in propagate) to visualize results (e.g., using `History.plot_line` if relevant), rather than interfacing with matplotlib directly.
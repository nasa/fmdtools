---
marp: true
---
![fmdtools logo](/docs/figures/logo.png)

# Intro to resilience modelling, simulation, and visualization in Python with fmdtools.
## Author: Daniel Hulse 
## Version: 2.0-alpha

-----------------

# Overview

- **Overview of fmdtools**
    - Purpose
    - Project Structure
    - Common Classes/Functions
    - Basic Syntax
- **Coding Activity**
    - Example model: [`examples/pump/ex_pump.py`](../examples/pump/ex_pump.py)
    - Workbook: [`examples/pump/Tutorial_unfilled.ipynb`](../examples/pump/Tutorial_unfilled.ipynb)
        - Model Instantiation
        - Simulation
        - Visualization/Analysis

-----------------
# Prerequisites

- Ideally, some pre-existing Python and Git knowledge
- Anaconda distribution 
    - Ideally this is already set up!
    - Download/install from: https://www.anaconda.com/products/individual
- A git interface
    - [Github Desktop](https://desktop.github.com/) (graphical git environment)
    - [git-scm](https://git-scm.com/) (stand-alone CLI)

-----------------
# Motivation: Modelling System Resilience

Resilience means taking a **dynamic understanding of risk and safety**

![resilience idea](/docs/figures/resilience_idea.png)

-----------------
# Why is Resilience Important?

![resilience importance](/docs/figures/resilience_importance.png)

-----------------
# Enabling proactive design process

![width:900px](/docs/figures/resilience_design.png)

- Especially relevant to **new systems** when **we don’t have data** 

-----------------

# Why fmdtools? Possible Competitors:

- Uncertainty Quantification tools:  (e.g. OpenCossan)
    - Doesn’t incorporate fault modelling/propagation/visualization aspects
- MATLAB/modelica/etc. Fault Simulation tools
    - Rely on pre-existing model/software stack--Useful, but often difficult to hack/extend (**not open-source**)
- Safety Assessment tools: (e.g. Alyrica, Hip-Hops)
    - Focused on quantifying safety, not necessarily resilience 
    - As a result, use **different model formalisms**!

-----------------

# Why fmdtools? Pros:

- Highly Expressive, modular model representation.
    - faults from any component can propagate to any other connected component via **undirected propagation**
    - highly-extensible code-based behavior representation
    - class structure enables **complex models** representing human behavior and systems of systems
- Research-oriented:
    - Written in/relies on the Python stack
    - Open source/free software
- Enables design:
    - Models can be parameterized an optimized!
    - Plug-and-play analyses and visualizations

-----------------

# Why **not** fmdtools? Cons:

- You already have a pre-existing system model
    - fmdtools models are built in fmdtools
    - if you have a simulink/modelica model, you may just want to use built-in tools

- You want to use this in production 
    - fmdtools is Class E Software and thus mainly suitable for research (or, at least, we don't gaurantee it)
    - Somewhat dynamic development history

-----------------

# What is fmdtools? A Python package for **design**, **simulation**, and **analysis** of resilience.

![module organization width:990px](/docs/figures/module_organization.svg)

-----------------

# What is fmdtools? Repo Structure


Repository (https://github.com/nasa/fmdtools/)
- `/fmdtools`: installable package
- `/examples`: example models with demonstrative notebooks and tests
- `/docs`: resources for [documentation](https://nasa.github.io/fmdtools/)
- `/tests`: stand-alone tests (and testing rigs)
- `README.md`: Basic package description 
- `CONTRIBUTORS.md`: Credit for contributions
- `requirements.txt`: List of requirements
- ... and other configuration files

-----------------

# Activity: Download and Install fmdtools

- repo link: https://github.com/nasa/fmdtools/
- set up repo:
    - create `path/to/fmdtools` folder for repo 
        - (usually in `/documents/GitHub`)
    - clone git into folder: 
        - `git clone https://github.com/nasa/fmdtools.git`
        - can also use webpage
- package installation: 
    - Open Python from anaconda (e.g., open Spyder)
    - Install with `pip install -e /path/to/fmdtools`

-----------------

# Analysis Workflow/Structure

![Analysis Workflow](/docs/figures/workflow.png)

-----------------

# Defining a Model

- What do we want out of a model?
    - What behaviors and how much fidelity do we need?
    - What functions/components and interactions make up the system?
        - Single function or multiple functions?
        - Is it controlled? Are there multiple agents?
- What type of simulation do we want to run?
    - Single-timestep vs multi-timestep vs network 
- What scenarioss do we want to study and how?
    - Failure modes and faulty behaviors
    - Disturbances and changes in parameters
    - What are the possible effects of hazards and how bad are they? 
        - By what metrics?

-----------------

# Defining a Model

![formalism example](/docs/figures/formalism_example.png)


-----------------
# Demo Model Activity: examples/pump/ex_pump.py

Notice the definitions and structure:
- **States**: `WaterStates`, `EEStates`, `SignalStates`
- **Flows**: `Water`, `EE`, `Signal`
- **Functions**: `ImportEE`, `ImportWater`, `ExportWater`, `MoveWater`, `ImportSignal`
    - **Flows**
    - **Modes** (e.g., `ImportEEMode`, `ImportSigMode`)
        - Mode probability model
        - Actual modes in `faultparams` entry
    - others attributes, e.g., `Timer`
- **Model**: `Pump` connects functions, flows, and defines `end_classification`
- **Parameter**: `PumpParam` defines values we can change in the simulation

-----------------

# More Resources for Model Definition

- Note the docs for model definition in https://nasa.github.io/fmdtools/docs/fmdtools.define.html

- Other examples also can be helpful: https://nasa.github.io/fmdtools/docs/Examples.html

-----------------

# Notebook Activity:

Open `/examples/pump/Tutorial_unfilled.ipynb`:
- Instantiate the model
    - `mdl = Pump()`
- Explore structure
    - Try different parameters! 
    - Change things!
    What does the model directory look like? 
    - `dir(mdl)`

-----------------

# Simulation Concepts: Static/Undirected Propagation

![Static Propagation](/docs/figures/propagation.png)

In a single timestep:
- Functions with `static_behavior()` methods simulate until behaviors converge (i.e., no new state values)
- Functions with `dynamic_behavior()` run once in defined order

-----------------

# Simulation Concepts: Propagation over Time

![Dynamic Propagation](/docs/figures/propagationovertime.png)

- Model increments (simulated + history updated) over each time-step until a **defined final time-step** or **specified indicator returns true**. 


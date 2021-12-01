<p align="center">
  <img src="docs/figures/logo.png" \>
</p>


fmdtools (Fault Model Design tools) is a toolkit for modelling system resilience in the early design phase. With it, one can simulate the effects of faults in a system to build resilience into the system design at a high level.  To achieve this, fmdtools provides a Python-based *design environment* where one can represent the system in a model, simulate the resilience of the model to faults, and analyze the resulting model responses to iteratively improve the resilience of the design.

This release is version 0.7.7.

[![DOI](https://zenodo.org/badge/212862445.svg)](https://zenodo.org/badge/latestdoi/212862445)

##### Key Features:

- fmdtools uses an object-oriented undirected graph-based model representation which enables arbitrary propagation of flow states through a model graph. As opposed to a *procedural* *directed* graph-based model representation (a typical strategy for developing fault models in code in which each function or component is represented by a method, the inputs and outputs are which are connected with connected functions/components in a larger model method), this enables one to:
  - propagate behaviors in multiple directions in a model graph, e.g., closing a valve will not just reduce flow in the downstream pipe but also increase pressure in upstream pipes.
  - define the data structures defining a function/component (e.g. states, faults, timed events) with the behavioral methods in a single logical chunk that can be re-used and modified for similar components and methods (that is, a class, instead of a set of unstructured variables and methods)

- fmdtools can represent the system at varying levels of fidelity through the design process so that one can start with a simple model and analysis and make it more detailed as the design is elaborated. A typical process of representing the system (from less to more detail) would involve:
  - Creating a network representation of the model functions and flows to visualize the system and identify structurally-important parts of the model's causal structure
  - Elaborating the flow attributes and function failure logic in a static propagation to simulate the timeless effects of faults in the model
  - Adding dynamic states and behaviors to the functions as well as a simulation times and operational phases in a dynamic propagation model to simulate the dynamic effects of faults simulated during different time-steps
  - Instantiating functions with component architectures to compare the expected resilience of each

- fmdtools provides convenience methods for quickly visualizing the results of fault simulations with commonly-used Python libraries to enable one to quickly assess:
  - effects of faults on functions and flows in the model graph at a given time-step
  - the behavior of system states over time in nominal and faulty scenarios
  - the effect of model input parameters (e.g., ranges, stochastic inputs) on nominal/faulty operations
  - the high-level results of a set of simulations in an FMEA-style table of faults, effects, rates, costs, and overall risk
  - fault injection times, responses, and weightings of a fault injection approach

In the future, we would like to add features for optimization/design exploration, non-deterministic fault propagation, and parallelism.

Finally, fmdtools is a research code and is under active development. As a result, Some use-cases may not work as desired and may change. If you find a bug or would like to contribute, contact the contributors.

----
## Getting Started

A version of the fmdtools toolkit can also be installed directly from the [PyPI package repository](https://pypi.org/project/fmdtools/) using `pip install fmdtools`. 

### Prerequisites

fmdtools requires Python 3 and depends on these packages:

- `networkx`
- `numpy`
- `ordered-dict`
- `dill`
- `pickle`
- `tqdm`
- `matplotlib`
- `pandas`
- `netgraph`

These packages are optional but reccomended:
- `jupyter notebook`	(for notebooks)
- `multiprocessing`     (for parallel execution--Pathos and multiprocess can also be used)
- `graphviz`			(to plot using graphviz options)
- `quadpy` 				(for quadrature sampling)
- `scipy` 				(for using statistical distributions in parameter sampling)
- `pyvis`				(for interactive html views of model graphs)
- `ffmpeg` 				(for animations)
- `shapely`				(for multirotor model)
- `pycallgraph` 		(for model profiling)

These must be installed (e.g. using `pip install packagename` or `conda install packagename`) them before running any of the codes in the repository. 

While it is not required to use any of the methods, Jupyter notebook is helpful for and for documenting simulation and analysis of a pre-existing model and required to follow through the provided examples.

### Documentation and Examples

An overview of fmdtools is provided in the accomanying paper:

[Hulse, D., Walsh, H., Dong, A., Hoyle, C., Tumer, I., Kulkarni, C., & Goebel, K. (2021). fmdtools: A Fault Propagation Toolkit for Resilience Assessment in Early Design. International Journal of Prognostics and Health Management, 12(3).](https://doi.org/10.36001/ijphm.2021.v12i3.2954)

Additionally, this repo provides a few resources to get familiar with fmdtools:

- Workshop slides and tutorial in `docs.Intro to fmdtools.pptx`, `pump example/ex_pump.py`, and `pump example/tutorial_complete.py`
- A shorter overview the toolkit methods and structure is provided in `docs/overview`.
- Some documented examples are provided, including:
  - A baseline example of most provided methods in conceptual design-stage pump system in `pump example/ex_pump.py` and `pump example/Pump Example Notebook.ipynb`
  - An case study following the modelling process of going from a less detailed to more detailed model is provided in `multirotor example\paper demonstration\Demonstration.ipynb` for the design of a multirotor drone.
  - An example replicating previous the simple electric power system implemented in [IBFM](https://github.com/DesignEngrLab/IBFM) in the `eps example` directory, with some basic fault propagation and visualization.
  - An example of modelling a dynamical system (without faults) is shown in the `\pandemic example` directory, with a reference stand-alone model is provided in `simple_model.py` and `fmd_model.py` and `fmd_model_script.py` showing how one might implement a distributed version of this model in `fmdtools`.
  - Using the `component` class to model human interactions with the modelled system in `hold-up tank example`.

- More detailed documentation for each of the classes/methods/modules is provided and can be viewed by going through the fmdtools source code (or by using `help(methodname)`)

----
## Contributors

[Daniel Hulse](https://github.com/hulsed)

[Hannah Walsh](https://github.com/walshh) : Network analysis codes

Sequoia Andrade : Graph visualization graphviz options

[Arpan Biswas](https://github.com/arpanbiswas52) : Multirotor optimization codes

[Hongyang Zhang](https://github.com/zhangho2) : Pandemic model

----
## License

MIT

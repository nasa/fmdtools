
**fmdtools** (Fault Model Design tools) is a toolkit for modelling system resilience in the early design phase. With it, one can simulate the effects of faults in a system to build resilience into the system design at a high level.  To achieve this, fmdtools provides a Python-based *design environment* where one can represent the system in a model, simulate the resilience of the model to faults, and analyze the resulting model responses to iteratively improve the resilience of the design.

Overview
====================================

The main impetus for the development of the fmdtools project was a lack existing tools to enable early function-based fault simulation for early functional hazard assessment. Researchers thus had to re-implement modelling, simulation, and analysis approaches for each new case study or methodological improvement. The fmdtools resolves this problem by separating resilience modelling, simulation, and analysis constructs from the model under study, enabling reuse of methodology between case studies. Towards this end, the fmdtools package provides three major pieces of functionality:

1. Model definition constructs which enable systematic early specification of the high level structure and behaviors of a system with concise syntax (fmdtools.modeldef).

2. Simulation methods which enable the quantification of system performance and propagation of hazards over a wide range of operational scenarios over a wide range of model types (fmdtools.faultsim).

3. Analysis methods for quantifying resilience and summarizing and visualizing behaviors and properties of interest (fmdtools.resultdisp).

Key Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fmdtools was developed with a number of unique features that differentiate it from existing safety/resilience simulation tools. 

- fmdtools uses an object-oriented undirected graph-based model representation which enables arbitrary propagation of flow states through a model graph. As opposed to a *procedural* *directed* graph-based model representation (a typical strategy for developing fault models in code in which each function or component is represented by a method, the inputs and outputs are which are connected with connected functions/components in a larger model method), this enables one to:
  
  - propagate behaviors in multiple directions in a model graph, e.g., closing a valve will not just reduce flow in the downstream pipe but also increase pressure in upstream pipes.
  
  - define the data structures defining a function/component (e.g. states, faults, timed events) with the behavioral methods in a single logical structure that can be re-used and modified for similar components and methods (that is, a class, instead of a set of unstructured variables and methods)

- fmdtools can represent the system at varying levels of fidelity through the design process so that one can start with a simple model and analysis and make it more detailed as the design is elaborated. A typical process of representing the system (from less to more detail) would involve:
  
  - Creating a network representation of the model functions and flows to visualize the system and identify structurally-important parts of the model's causal structure
  
  - Elaborating the flow attributes and function failure logic in a static propagation to simulate the timeless effects of faults in the model
  
  - Adding dynamic states and behaviors to the functions as well as a simulation times and operational phases in a dynamic propagation model to simulate the dynamic effects of faults simulated during different time-steps
  
  - Instantiating functions with component architectures to compare the expected resilience and behaviors of each
  
  - Defining stochastic behavioral and input parameters to simulate and analyze system resilience throughout the operational envelope

- fmdtools provides convenience methods for quickly visualizing the results of fault simulations with commonly-used Python libraries to enable one to quickly assess:
  
  - effects of faults on functions and flows in the model graph at a given time-step
  
  - the behavior of system states over time in nominal and faulty scenarios over a range of operational parameters
  
  - the effect of model input parameters (e.g., ranges, stochastic inputs) on nominal/faulty operations
  
  - the high-level results of a set of simulations in an FMEA-style table of faults, effects, rates, costs, and overall risk
  
  - simulation responses over a range or distribution of model and scenario parameters


An overview of an earlier version of fmdtools (0.6.2) is provided in the paper:

`Hulse, D., Walsh, H., Dong, A., Hoyle, C., Tumer, I., Kulkarni, C., & Goebel, K. (2021). fmdtools: A Fault Propagation Toolkit for Resilience Assessment in Early Design. International Journal of Prognostics and Health Management, 12(3). <https://doi.org/10.36001/ijphm.2021.v12i3.2954>`_


fmdtools is a research code and is under active development. As a result, Some use-cases may not work as desired and may change. If you find a bug or would like to contribute, contact the contributors. 

Getting Started
====================================

The latest public version of fmdtools can be downloaded from the `fmdtools github repository <https://github.com/nasa/fmdtools/>`_ e.g., using:

::

   git clone https://github.com/nasa/fmdtools.git

A version of the fmdtools toolkit can also be installed directly from the `PyPI package repository <https://pypi.org/project/fmdtools/>`_ using ``pip install fmdtools``. 

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fmdtools requires Python 3 and depends directly on these packages (see requirements.txt):

::

   scipy
   tqdm
   multiprocessing
   networkx
   numpy
   matplotlib
   pandas
   ordered-set
   dill 
   pickle

These packages are optional but recommended to enable specific fmdtools use-cases and to work with examples in the repository:

::

   jupyter notebook			#(for repository notebooks)
   graphviz				#(to plot using graphviz options)
   pyvis				#(for interactive html views of model graphs)
   quadpy 				#(for quadrature sampling)
   ffmpeg 				#(for animations)
   shapely				#(for multirotor model)
   pycallgraph2				#(for model profiling)

These must be installed (e.g. using ``pip install packagename`` or ``conda install packagename``) them before running any of the codes in the repository. 


Contributions
====================================

fmdtools is developed primarily by the `Resilience Analysis and Design <https://ti.arc.nasa.gov/tech/rse/research/rad/>`_ research project. External contributions are welcome under a Contributor License Agreement:

`Individual CLA <fmdtools_Individual_CLA.pdf>`

`Corporate CLA <fmdtools_Corporate_CLA.pdf>`

Contributors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Daniel Hulse <https://github.com/hulsed>`_ : Primary Author and point-of-contact

`Hannah Walsh <https://github.com/walshh>`_ : Network Analysis Codes

`Sequoia Andrade <https://ti.arc.nasa.gov/profile/andrade/>`_ : Graph visualization graphviz options, Code review



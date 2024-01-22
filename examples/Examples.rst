Examples
==============================================

This repository provides several examples models in the `/examples` directory, with some corresponding documentation in the form of notebooks (which may be accessed here). 

Pump
+++++++++++++++++++++++++++++++++++++++++++++

The pump example model is located in `examples/pump/ex_pump.py`, which is then used in the following notebooks:

- **Tutorial** which was designed to provide a basic interactive tutorial for learning basic fmdtools functions (along with the `Intro to fmdtools <docs/Intro_to_fmdtools.md>`_ workshop). This tutorial has two components: 

  - `Tutorial_unfilled.ipynb <../examples/pump/Tutorial_unfilled.ipynb>`_, which is to be used by students, and has the basic structure without the code filled in.

  - `Tutorial_complete.ipynb <../examples/pump/Tutorial_complete.ipynb>`_, which is to be used by the instructor, and has the basic code filled in to follow along with.
  
- `Pump Example Notebook <../examples/pump/Pump_Example_Notebook.ipynb>`_ is helpful for understanding the breadth of fmdtools plotting, tabulation, and visualization capabilities. It covers:
  
  - A variety of graphing use-cases in :mod:`fmdtools.analyze.graph` functions which enable viewing different graph types, simulation results at individual times, and overall model statistics/results.
  
  - :mod:`fmdtools.analyze.tabulate` functions for viewing simulation results over time and summarizing run information.
  
  - Saving/Loading `fmdtools.analyze.result.Result` data structures.
  
- `Using Parallel Computing in fmdtools <../examples/pump/Parallelism_Tutorial.ipynb>`_ covers how to reduce computational costs for computationally-expensive simulation use-cases (e.g., sampling large numbers of fault scenarios or running complex models with large numbers of timesteps). It covers:

  - Using process pools as arguments to :mod:`fmdtools.sim.propagate` methods to speed up simulation. A comparison of process pools from different external python packages are provided.
  
  - Different options for history tracking and staged execution which can reduce computational costs when desired.
  
  - Profiling models with ``cProfile`` to discover what parts are most computationally-expensive.

- `Optimization <../examples/pump/Optimization.ipynb>`_, shows some of the basics of working with :class:`fmdtools.sim.search.DisturbanceProblem` and :class:`fmdtools.sim.search.ParameterSimProblem` classes for optimization. 

In addition to `ex_pump.py`, more use-cases are demonstrated in the derivative models `pump_indiv.py` (which shows how :mod:`fmdtools.define.block.function` objects can be simulated individually outside of a model) and `pump_stochastic.py` (which demonstrates stochastic modeling in fmdtools). Stochastic Modeling is further demonstrated in the notebook:


- `Stochastic Modeling in fmdtools <../examples/pump/Stochastic_Modelling.ipynb>`_ , which covers defining and simulating stochastic models--models with random internal behaviors. This includees:

  - Setting up random states in functions using :class:`fmdtools.define.container.rand.Rand` and :meth:`fmdtools.define.container.rand.Rand.set_rand_state()`.
  
  - Simulating stochastic models using the `run_stochastic` parameter in :mod:`fmdtools.sim.propagate` functions, as well as setting up a :class:`fmdtools.sim.sample.ParameterSample` with multiple seeds to run a set of random simulations.
  
  - Using :meth:`fmdtools.analyze.history.History.plot_line()` to visualize the results of multiple stochastic simulations over time, and analyze quantities of interest using :class:`fmdtools.analyze.tabulate.FMEA`, :class:`fmdtools.analyze.tabulate.Comparison()`

- `AST Sampling <../examples/pump/AST_Sampling.ipynb>`_, shows how fmdtools models called from the `AdaSress Julia <https://www.nasa.gov/content/tech/rse/research/adastress>_` package to leverage the adaptive stress testing methodology using the :class:`fmdtools.sim.search.DynamicInterface` class.


.. toctree::
   :hidden:
   
   ../examples/pump/Tutorial_complete.ipynb
   ../examples/pump/Pump_Example_Notebook.ipynb
   ../examples/pump/Parallelism_Tutorial.ipynb
   ../examples/pump/Optimization.ipynb
   ../examples/pump/Stochastic_Modelling.ipynb
   ../examples/pump/AST_Sampling.ipynb

Multirotor
+++++++++++++++++++++++++++++++++++++++++++++

The multirotor example model has several models of drones modeled at differing levels of detail which are then used in the following example notebooks;

- `fmdtools Paper Demonstration <../examples/multirotor/Demonstration.ipynb>`_  is helpful for understanding how a model can be matured as more details are added, covering:

  - The :mod:`fmdtools.analyze.graph` module.
  
  - Basic simulation of dynamic and static models using methods in :mod:`fmdtools.sim.propagate` and usage of class :class:`fmdtools.sim.sample.SampleApproach` for fault sampling
  
  - Analysis using Basic analysis/results processing capabilities
 
- The `Urban Drone Model <../examples/multirotor/drone_mdl_urban.py>`_ is helpful for understanding how to set up gridworlds using :class:`fmdtools.define.object.coords.Coords` and an Environment class. `Urban Drone Demo <../examples/multirotor/Urban_Drone_Demo.ipynb>`_ demonstrates how this gridworld can be used in simulation.
 
- `Multirotor Optimization <../examples/multirotor/Multirotor_Optimization.ipynb>`_ shows how the design, operations, and contingency management of a system can be co-optimized with the :class:`fmdtools.sim.search.ProblemArchitecture` class. 

- The support files include various implementations of the drone model.

  - `drone_mdl_static.py` is the baseline model of the drone for static modeling that is used in the other models.

  - `drone_mdl_dynamic.py` expands on the static model to allow for dynamic simulation. It generates behavior-over-time graphs and dynamic/phase-based FMEAs. 

  - `drone_mdl_hierarchical.py` is used to compare system architectures. First by seeing how faults effect the behaviors in each architecture, then by seing how it affects the overall system resilience.

  - `drone_mdl_opt.py` is a modified version of the hierarchical done that encompasses autonomous path planning, rotors, electrical system, and control of the drone. It is parameterized with the following parameters: The rotor and battery architecture can be changed, the flight height can be changed to support different heights, which in turn changes the drone's flight plan, and there is now a `ManageHealth` function which reconfigures the flight depending on detected faults.


.. toctree::
   :hidden:
   
   ../examples/multirotor/Demonstration.ipynb
   ../examples/multirotor/Urban_Drone_Demo.ipynb
   ../examples/multirotor/Multirotor_Optimization.ipynb

Tank
+++++++++++++++++++++++++++++++++++++++++++++

The tank example is a fairly simple model of a tank, inlet valve, and outlet valve. This example is shown in the following notebooks.

- `Hold-up Tank Model <../examples/tank/Tank_Analysis.ipynb>`_ uses the :class:`fmdtools.sim.sample.SampleApproach` class to model human interactions with the modeled system (in `tank_model.py`).

- `Tank Optimization <../examples/tank/Tank_Optimization.ipynb>`_ shows how design and contingency management of a system (in `tank_optimization_model.py`) can be co-optimized with the :class:`fmdtools.sim.search.ProblemArchitecture` class, as well as external solvers.

The support files include various implementations of the tank model.


- The baseline Tank Model (`tank_model.py`), a dynamical implementation of a human-operated tank system to show how fmdtools can be used to model human errors.

- A demonstration Optimization Tank Model (`tank_opt.py`), resilience optimization architectures using :class:`fmdtools.sim.search.ProblemArchitecture`. 
 
- The main Tank optimization model (`tank_optimization.py`) similar to `tank_model` and `tank_opt` and is a dynamical implementation of a tank system with contingency management. 

.. toctree::
   :hidden:
   
   ../examples/tank/Tank_Analysis.ipynb
   ../examples/tank/Tank_Optimization.ipynb


EPS
+++++++++++++++++++++++++++++++++++++++++++++

The EPS model is a model of a simple electric power system in `eps.py`, which shows how undirected propagation can be used in a simple static (i.e., one time-step) modeling use-case. 

- `EPS Example Notebook <../examples/eps/EPS_Example_Notebook.ipynb>`_ demonstrates this model and some basic fmdtools methods. It shows how fmdtools can be used for purely static propagation models where dynamic states are not a concern. This is a replication of a previous simple electric power system implemented in `IBFM <https://github.com/DesignEngrLab/IBFM>_`.

-  `The supporting EPS Model `eps.py` builds the eps model for the notebook. The main purpose of this system is to supply power to optical, mechanical, and heat loads. Failure behavior of the system is represented at a high level using solely the functions of the system. 

.. toctree::
   :hidden:
   
   ../examples/eps/EPS_Example_Notebook.ipynb

Action Sequence Graph
+++++++++++++++++++++++++++++++++++++++++++++

`Action Sequence Graph <../examples/eps/Action_Sequence_Graph.ipynb>`_ shows a very basic example of using the :class:`fmdtools.define.architecture.action.ActionArchitecture` structure to define sequences of actions or modes, which is helpful for modeling human and autonomous behaviors.

.. toctree::
   :hidden:
   
   ../examples/asg_demo/Action_Sequence_Graph.ipynb

Rover
+++++++++++++++++++++++++++++++++++++++++++++

The Rover model showcases more advanced methodologies that can be used in fmdtools, and has essentially been the developers' demo case study for advancing the state-of-the-art in resilience simulation. These demonstrations include:

- `Rover Setup Notebook <../examples/rover/Rover_Setup_Notebook.ipynb>`_ , which introduces the model and basic usages.

- `Defining and Visualizing fmdtools Model Structures <../examples/rover/Model_Structure_Visualization_Tutorial.ipynb>`_ is helpful for understanding how a given model simulates over time. It covers the class  :class:`fmdtools.analyze.graph.Graph` and containing methods.

- `Defining Parameter Samples in fmdtools <../examples/rover/ParameterSample_Use-Cases.ipynb>`_ , which covers simulating the model at different parameters in nominal/faulty scenarios. This includes:

  - Setting up a nominal parameter sampling approach using :class:`fmdtools.sim.sample.ParameterSample` and simulating it with :func:`fmdtools.sim.propagate.parameter_sample()` and :func:`fmdtools.sim.propagate.nested_sample()` methods for nominal and faulty simulations.

  - Using analysis classes like :func:`fmdtools.analyze.tabulate.NominalEnvelope` and :func:`fmdtools.analyze.tabulate.Comparison` to visualize quantities of interest for the simulation over a range of nominal parameters.
  
- `Defining Fault Sampling Approaches in fmdtools <../examples/rover/Approach_Use-Cases.ipynb>`_ covers how to set up a fault sampling approach and use it to simulate a large number of hazardous scenarios in a model. This includes:

  - Adding fault and operational modes to Model functions using the class :class:`fmdtools.define.container.mode.Mode` and explanation of the `phases` and `exclusive` options.
  
  - Using :func:`fmdtools.analyze.phases.from_hist` to setting a :class:`fmdtools.sim.sample.SampleApproach` up which samples individual faults based on the phases of the model and/or function defined by their operational modes.

  - Using :func:`fmdtools.analyze.phases.PhaseMap` to visualize the phases and modes of a model over time and to visualize the consequences of each fault scenario in the approach within each phase.
  
  - Using the `defaultsamp` option to control how many time-steps in each phase are sampled (and when).

- `HFAC Analyses <../examples/rover/HFAC_Analyses.ipynb>`_ and `IDETC_Human_Paper_Analysis <../examples/rover/IDETC_Human_Paper_Analysis.ipynb>`_ showcase fmdtools human factors modeling capabilities (action sequence graphs, performance shaping factors, etc.).
  
- `Degradation Modelling Notebook <../examples/rover/degradation_modelling/Degradation_Modelling_Notebook.ipynb>`_ shows how degradation models can be used to sample resilience model parameters.

- `Rover Mode Notebook <../examples/rover/fault_sampling/Rover_Mode_Notebook.ipynb>`_ shows how fault scenarios can be exhaustively generated to augment hazard identification.
  
- `Search_Comparison <../examples/rover/optimization/Search_Comparison.ipynb>`_ shows an (in-development) method for searching the faulty state-space to find a unique set of hazardous scenarios. 
  
- `Rover Response Optimization <../examples/rover/optimization/Rover_Response_Optimization.ipynb>`_ further demonstrates the optimization of parameters over a set of fault scenarios using :class:`fmdtools.sim.search.ProblemArchitecture`.
  
- The support files include various implementations of the rover model.
  - `rover_model.py` defines the functions and flows used in the analysis.
  - `rover_degradation.py` extends the model to include degradated states.
  - `rover_model_human.py` extends the model to include human faults and response.


.. toctree::
   :hidden:
   
   ../examples/rover/Rover_Setup_Notebook.ipynb
   ../examples/rover/Mode_Structure_Visualization_Tutorial.ipynb
   ../examples/rover/Approach_Use-Cases.ipynb
   ../examples/rover/Nominal_Approach_Use-Cases.ipynb
   ../examples/rover/HFAC_Analyses/HFAC_Analyses.ipynb
   ../examples/rover/HFAC_Analyses/IDETC_Human_Paper_Analysis.ipynb
   ../examples/rover/degradation_modelling/Degradation_Modelling_Notebook.ipynb
   ../examples/rover/fault_sampling/Rover_Mode_Notebook.ipynb
   ../examples/rover/optimization/Search_Comparison.ipynb
   ../examples/rover/optimization/Rover_Response_Optimization.ipynb

Multiflow Demo
+++++++++++++++++++++++++++++++++++++++++++++

The multiflows example is limited to the model in the file `multiflow_demo.py`, which shows basic usage of the :class:`fmdtools.define.flow.multiflow.MultiFlow` and :class:`fmdtools.define.flow.commsflow.CommsFlow` flow structures. `Multiflow and Commsflow Demonstration <..examples/multiflow_demo/MultiFlow_and_Commsflow_Demonstation.ipynb>`_ is a tutorialized version of that file discussing the step by step creation and use of :class:`fmdtools.analyze.graph.FunctionArchitectureTypeGraph` to view model types.

.. toctree::
  :hidden:

  ../examples/multiflow_demo/Multiflow_and_Commsflow_Demonstration.ipynb

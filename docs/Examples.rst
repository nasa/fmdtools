Examples
==============================================

This repository provides several resources to get familiar with fmdtools. 

To get started with fmdtools, it is reccomended to start with the **Intro to fmdtools** workshop, which provides a basic high-level overview of modelling and simulation in fmdtools, with a corresponding model and example notebook to fill out. To start the workshop, first download the workshop slides (:download:`Intro to fmdtools.pptx <docs/Intro to fmdtools.pptx>`), tutorial model (:download:`ex_pump.py <example_pump/ex_pump.py>`), and Unfilled Notebook :download:`ex_pump.py <example_pump/Tutorial_unfilled.ipynb>`.  If you cloned fmdtools, you can just navigate to these files in the repository--they are in a directory called ``example_pump``. Then, follow along with the slides and the (see: `filled-in notebook <example_pump/Tutorial_complete.ipynb>`_).
   
After completing the workshop, it can be helpful to run through the following notebooks to better understand fmdtools modelling, simulation, and analysis basics as well as more advanced features and use-cases:

- `Defining and Visualizing fmdtools Model Structures <docs/Model_Structure_Visualization_Tutorial.ipynb>`_ is helpful for understanding how a given model simulates over time. It covers the methods:

  - :func:`fmdtools.analyze.graph.show()`

  - :func:`fmdtools.analyze.graph.exec_order()`

  - :func:`fmdtools.analyze.plot.graph_order()`

- `fmdtools Paper Demonstration <example_multirotor/Demonstration.ipynb>`_  is helpful for understanding network/static/dynamic/hierarchical model types, covering:

  - The :mod:`fmdtools.sim.networks` module.
  
  - Basic simulation of dynamic and static models using methods in :mod:`fmdtools.sim.propagate` and usage of class :class:`fmdtools.define.SampleApproach` for fault sampling
  
  - Analysis using Basic analysis/results processing capabilities

- `Pump Example Notebook <example_pump/Pump_Example_Notebook.ipynb>`_ is helpful for understanding the breadth of fmdtools plotting, tabulation, and visualization capabilities. It covers:
  
  - A variety of graphing usecases in :mod:`fmdtools.analyze.graph` functions which enable viewing different graph types, simulation results at individual times, and overall model statistics/results.
  
  - :mod:`fmdtools.analyze.tabulate` functions for viewing simulation results over time and summarizing run information
  
  - Save/load using the `dill` package

- `Defining Fault Sampling Approaches in fmdtools <docs/Approach_Use-Cases.ipynb>`_ covers how to set up a fault sampling approach and use it to simulate a large number of hazardous scenarios in a model. This includes:

  - Adding fault and operational modes to Model functions using the method :meth:`fmdtools.define.Block.assoc_modes()` and explanation of the `key_phases_by` and `exclusive` options.
  
  - Using :func:`fmdtools.sim.result.History.get_modephases()` to setting a :class:`fmdtools.define.SampleApproach` up which samples individual faults based on the phases of the model and/or function defined by their operational modes.

  - Using :func:`fmdtools.analyze.plot.phases()` to visualize the phases and modes of a model over time and :func:`fmdtools.analyze.plot.samplecosts()` to visualize the consequences of each fault scenario in the approach within each phase.
  
  - Using the `defaultsamp` option and :meth:`fmdtools.define.SampleApproach.prune_scenarios()` in :class:`fmdtools.define.SampleApproach` to control how many time-steps in each phase are sampled (and when).
  
- `Using Parallel Computing in fmdtools <example_pump/Parallelism_Tutorial.ipynb>`_ covers how to reduce computational costs for computationally-expensive simulation use-cases (e.g., sampling large numbers of fault scenarios or running complex models with large numbers of timesteps). It covers:

  - Using process pools as arguments to :mod:`fmdtools.sim.propagate` methods to speed up simulation. A comparison of process pools from different external python packages are provided.
  
  - Different options for history tracking and staged execution which can reduce computational costs when desired.
  
  - Profiling models with ``cProfile`` and ``pycallgraph2`` to discover what parts are most computationally-expensive.

- `Defining Nominal Approaches in fmdtools <docs/Nominal_Approach_Use-Cases.ipynb>`_ , which covers simulating the model at different parameters in nominal/faulty scenarios. This includes:

  - Setting up a nominal parameter sampling approach using :class:`fmdtools.define.NominalApproach` and simulating it with :func:`fmdtools.sim.propagate.nominal_approach()` and :func:`fmdtools.sim.propagate.nested_approach()` methods for nominal and faulty simulations.

  - Using analysis functions like :func:`fmdtools.analyze.tabulate.nominal_vals_1d()` and :func:`fmdtools.analyze.plot.nominal_factor_comparison()` to visualize quantities of interest for the simulation over a range of nominal parameters.

  - Using analysis functions like :func:`fmdtools.analyze.tabulate.resilience_factor_comparison()` and :func:`fmdtools.analyze.plot.resilience_factor_comparison()` to visualize resilience metrics of the model to a set of fault modes over a range of nominal parameters.

- `Stochastic Modelling in fmdtools <example_pump/Stochastic_Modelling.ipynb>`_ , which covers defining and simulating stochastic models--models with random internal behaviors. This includees:

  - Setting up random states in functions using :meth:`fmdtools.define.Block.assoc_rand_state()`, :meth:`fmdtools.define.Block.set_rand()`, and :meth:`fmdtools.define.Block.to_default()`.
  
  - Simulating stochastic models using the `run_stochastic` parameter in :mod:`fmdtools.sim.propagate` functions, as well as setting up a :class:`fmdtools.define.NominalApproach` with multiple seeds to run a set of random simulations.
  
  - Using :func:`fmdtools.analyze.plot.mdlhists()` to visualize the results of multiple stochastic simulations over time, and analyze quantities of interest using :func:`fmdtools.analyze.tabulate.nested_stats()`, :func:`fmdtools.analyze.tabulate.resilience_factor_comparison()`

A number of more advanced techniques have been developed using the rover model in `example_rover`. `Rover Setup Notebook <example_rover/Rover Setup Notebook.ipynb>`_ , introduces this model, which is then used to:

  - Showcase fmdtools human factors modelling capabilities (action sequence graphs, performance shaping factors, etc.) in `HFAC Analyses <example_rover/HFAC Analyses.ipynb>`_ and `IDETC_Human_Paper_Analysis <example_rover/IDETC_Human_Paper_Analysis.ipynb>`_
  
  - Show how degradation models can be used to sample resilience model parameters in `Degradation Modelling Notebook <example_rover/degradation_modelling/Degradation Modelling Notebook.ipynb>`_

  - Show how fault scenarios can be exhaustively generated to augment hazard identification in `Rover Mode Notebook <example_rover/fault_sampling/Rover Mode Notebook.ipynb>`_
  
  - Show a method for searching the faulty state-space to find a unique set of hazardous scenarios in `Search Comparison <example_rover/optimization/Search Comparison.ipynb>`_ 
  
  - Demonstrate the optimization of parameters over a set of fault scenarios using :class:`ProblemInterface` in `Rover Response Optimization <example_rover/optimization/Rover Response Optimization.ipynb>`_ 

The following notebooks showcase the optimization interfaces in :mod:`fmdtools.sim.search`:

  - `Optimization <example_pump/Optimization.ipynb>`_ shows some of the basics of working with the :class:`fmdtools.sim.search.ProblemInterface` class for optimization. 

  - `AST Sampling <example_pump/AST_Sampling.ipynb>`_ shows how fmdtools models called from the AdaSress Julia package to leverage the adaptive stress testing methodology using the :class:`fmdtools.sim.search.DynamicInterface` class.
 
  - `Multirotor Optimization <example_multirotor/Multirotor Optimization.ipynb>`_ shows how the design, operations, and contingency management of a system can be co-optimized with the :class:`fmdtools.sim.search.ProblemInterface` class. 
  
  - `Tank Optimization <example_tank/Tank Optimization.ipynb>`_ shows how design and contingency management of a system can be co-optimized with the :class:`fmdtools.sim.search.ProblemInterface` class, as well as external solvers.

There are also two other example models which demonstrate specialized use-cases:

- `Hold-up Tank Model <example_tank/Tank_Analysis.ipynb>`_ uses the :class:`fmdtools.define.SampleApproach` class to model human interactions with the modelled system.

- `EPS Example Notebook <example_eps/EPS_Example_Notebook.ipynb>`_ shows a simple static modelling use-case.

.. toctree::
   :hidden:
   
   example_pump/Tutorial_complete.ipynb
   docs/Model_Structure_Visualization_Tutorial.ipynb
   example_multirotor/Demonstration.ipynb
   example_pump/Pump_Example_Notebook.ipynb
   docs/Approach_Use-Cases.ipynb
   example_pump/Parallelism_Tutorial.ipynb
   docs/Nominal_Approach_Use-Cases.ipynb
   example_pump/Stochastic_Modelling.ipynb
   example_tank/Tank_Analysis.ipynb
   example_eps/EPS_Example_Notebook.ipynb
   example_rover/Rover Setup Notebook.ipynb
   example_rover/HFAC Analyses.ipynb
   example_rover/IDETC_Human_Paper_Analysis.ipynb
   example_rover/degradation_modelling/Degradation Modelling Notebook.ipynb
   example_rover/fault_sampling/Rover Mode Notebook.ipynb
   example_rover/optimization/Rover Response Optimization.ipynb
   example_pump/Optimization.ipynb
   example_pump/AST_Sampling.ipynb
   example_multirotor/Multirotor Optimization.ipynb
   example_tank/Tank Optimization.ipynb


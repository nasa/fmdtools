Examples
==============================================

This repository provides several resources to get familiar with fmdtools. 

To get started with fmdtools, it is reccomended to start with the **Intro to fmdtools** workshop, which provides a basic high-level overview of modelling and simulation in fmdtools, with a corresponding model and example notebook to fill out. To start the workshop, first download the workshop slides (:download:`Intro to fmdtools.pptx <docs/Intro to fmdtools.pptx>`), tutorial model (:download:`ex_pump.py <example_pump/ex_pump.py>`), and Unfilled Notebook :download:`ex_pump.py <example_pump/Tutorial_unfilled.ipynb>`.  If you cloned fmdtools, you can just navigate to these files in the repository--they are in a directory called ``example_pump``. Then, follow along with the slides and the (see: `filled-in notebook <example_pump/Tutorial_complete.ipynb>`_).
   
After completing the workshop, it can be helpful to run through the following notebooks to better understand fmdtools modelling, simulation, and analysis basics as well as more advanced features and use-cases:

- `Defining and Visualizing fmdtools Model Structures <docs/Model_Structure_Visualization_Tutorial.ipynb>`_ is helpful for understanding how a given model simulates over time. It covers the methods:

  - :func:`fmdtools.resultdisp.graph.show()`

  - :func:`fmdtools.resultdisp.graph.exec_order()`

  - :func:`fmdtools.resultdisp.plot.graph_order()`

- `fmdtools Paper Demonstration <example_multirotor/Demonstration.ipynb>`_  is helpful for understanding network/static/dynamic/hierarchical model types, covering:

  - The :mod:`fmdtools.faultsim.networks` module.
  
  - Basic simulation of dynamic and static models using methods in :mod:`fmdtools.faultsim.propagate` and usage of class :class:`fmdtools.modeldef.SampleApproach` for fault sampling
  
  - Analysis using :func:`fmdtools.resultdisp.process.hists()`, :func:`fmdtools.resultdisp.plot.mdlhistvals()` :func:`fmdtools.resultdisp.tabulate.fullfmea()`, and :func:`fmdtools.resultdisp.tabulate.simplefmea()`

- `Pump Example Notebook <example_pump/Pump_Example_Notebook.ipynb>`_ is helpful for understanding the breadth of fmdtools plotting, tabulation, and visualization capabilities. It covers:
  
  - A variety of graphing usecases in :mod:`fmdtools.resultdisp.graph` functions which enable viewing different graph types, simulation results at individual times, and overall model statistics/results.
  
  - :mod:`fmdtools.resultdisp.tabulate` functions for viewing simulation results over time and summarizing run information
  
  - Save/load using the `dill` package

- `Defining Fault Sampling Approaches in fmdtools <docs/Approach_Use-Cases.ipynb>`_ covers how to set up a fault sampling approach and use it to simulate a large number of hazardous scenarios in a model. This includes:

  - Adding fault and operational modes to Model functions using the method :meth:`fmdtools.modeldef.Block.assoc_modes()` and explanation of the `key_phases_by` and `exclusive` options.
  
  - Using :func:`fmdtools.resultdisp.process.modephases()` to setting a :class:`fmdtools.modeldef.SampleApproach` up which samples individual faults based on the phases of the model and/or function defined by their operational modes.

  - Using :func:`fmdtools.resultdisp.plot.phases()` to visualize the phases and modes of a model over time and :func:`fmdtools.resultdisp.plot.samplecosts()` to visualize the consequences of each fault scenario in the approach within each phase.
  
  - Using the `defaultsamp` option and :meth:`fmdtools.modeldef.SampleApproach.prune_scenarios()` in :class:`fmdtools.modeldef.SampleApproach` to control how many time-steps in each phase are sampled (and when).
  
- `Using Parallel Computing in fmdtools <example_pump/Parallelism_Tutorial.ipynb>`_ covers how to reduce computational costs for computationally-expensive simulation use-cases (e.g., sampling large numbers of fault scenarios or running complex models with large numbers of timesteps). It covers:

  - Using process pools as arguments to :module:`fmdtools.faultsim.propagate` methods to speed up simulation. A comparison of process pools from different external python packages are provided.
  
  - Different options for history tracking and staged execution which can reduce computational costs when desired.
  
  - Profiling models with ``cProfile`` and ``pycallgraph2`` to discover what parts are most computationally-expensive.

- `Defining Nominal Approaches in fmdtools <docs/Nominal_Approach_Use-Cases.ipynb>`_ , which covers simulating the model at different parameters in nominal/faulty scenarios. This includes:

  - Setting up a nominal parameter sampling approach using :class:`fmdtools.modeldef.NominalApproach` and simulating it with :func:`fmdtools.faultsim.propagate.nominal_approach()` and :func:`fmdtools.faultsim.propagate.nested_approach()` methods for nominal and faulty simulations.

  - Using analysis functions like :func:`fmdtools.resultdisp.tabulate.nominal_vals_1d()` and :func:`fmdtools.resuldisp.plot.nominal_factor_comparison()` to visualize quantities of interest for the simulation over a range of nominal parameters.

  - Using analysis functions like :func:`fmdtools.resuldisp.tabulate.resilience_factor_comparison()` and :func:`fmdtools.resuldisp.plot.resilience_factor_comparison()` to visualize resilience metrics of the model to a set of fault modes over a range of nominal parameters.

- `Stochastic Modelling in fmdtools <example_pump/Stochastic_Modelling.ipynb>`_ , which covers defining and simulating stochastic models--models with random internal behaviors. This includees:

  - Setting up random states in functions using :meth:`fmdtools.modeldef.Block.assoc_rand_state()`, :meth:`fmdtools.modeldef.Block.set_rand()`, and :meth:`fmdtools.modeldef.Block.to_default()`.
  
  - Simulating stochastic models using the `run_stochastic` parameter in :mod:`fmdtools.faultsim.propagate` functions, as well as setting up a :class:`fmdtools.modeldef.NominalApproach` with multiple seeds to run a set of random simulations.
  
  - Using :func:`fmdtools.resultdisp.plot.mdlhists()` to visualize the results of multiple stochastic simulations over time, and analyze quantities of interest using :func:`fmdtools.resultdisp.tabulate.nested_stats()`, :func:`fmdtools.resultdisp.tabulate.resilience_factor_comparison()`


There are also two other example models which demonstrate specialized use-cases:

- `Hold-up Tank Model <example_tank/Tank_Analysis.ipynb>`_ uses the :class:`fmdtools.modeldef.SampleApproach` class to model human interactions with the modelled system.

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
Pump Model
---------------------------------------------

A simple pump model to demonstrate various capabilities of fmdtools.

Models
/////////////////////////////////////////////


- `ex_pump.py`: Main pump model file

- `pump_indiv.py`: Modelling of an individual pump Function outside the context of a FunctionArchitecture.

- `pump_stochastic.py`: Modelling of the pump with stochastic behaviors

Scripts and tests:
/////////////////////////////////////////////

- `parallelism_methods.py`: Some functions for exploring parallel model execution

- `test_pump_example`: Examples of tests one might run to verify simulation behavior/setup

- `test_pump_stochastic.py`: Tests of fmdtools using the stochastic pump model

- `test_pump.py`: Tests of fmdtools using the pump model

Notebooks
/////////////////////////////////////////////

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

- `AST Sampling <../examples/pump/AST_Sampling.ipynb>`_, shows how fmdtools models called from the `AdaSress Julia <https://www.nasa.gov/content/tech/rse/research/adastress>_` package to leverage the adaptive stress testing methodology using the :class:`fmdtools.sim.search.DynamicInterface` class.

- `Optimization.ipynb`: Showing how to use optimization methods in `fmdtools.sim.search` to optimize scenarios and responses.

- `Parallelism_Tutorial.ipynb`: Shows how to use parallelism in the context of simulation using the `multiprocessing` library.

- `Stochastic Modeling in fmdtools <../examples/pump/Stochastic_Modelling.ipynb>`_ , which covers defining and simulating stochastic models--models with random internal behaviors. This includees:

  - Setting up random states in functions using :class:`fmdtools.define.container.rand.Rand` and :meth:`fmdtools.define.container.rand.Rand.set_rand_state()`.
  
  - Simulating stochastic models using the `run_stochastic` parameter in :mod:`fmdtools.sim.propagate` functions, as well as setting up a :class:`fmdtools.sim.sample.ParameterSample` with multiple seeds to run a set of random simulations.
  
  - Using :meth:`fmdtools.analyze.history.History.plot_line()` to visualize the results of multiple stochastic simulations over time, and analyze quantities of interest using :class:`fmdtools.analyze.tabulate.FMEA`, :class:`fmdtools.analyze.tabulate.Comparison()`


.. toctree::
   :hidden:
   
   ../examples/pump/Tutorial_complete.ipynb
   ../examples/pump/Pump_Example_Notebook.ipynb
   ../examples/pump/Parallelism_Tutorial.ipynb
   ../examples/pump/Optimization.ipynb
   ../examples/pump/Stochastic_Modelling.ipynb
   ../examples/pump/AST_Sampling.ipynb
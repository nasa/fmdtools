Rover Model
---------------------------------------------

Very basic Action Sequence Graph model

Models
/////////////////////////////////////////////

- `rover_model.py` defines the functions and flows used in the analysis.

- `rover_degradation.py` extends the model to include degradated states.

- `rover_model_human.py` extends the model to include human faults and response.

Scripts and tests:
/////////////////////////////////////////////

- `opt_rover.py` defines some optimization setup(s) with the rover

- `rover_mode_space.py` is a script for testing synthetic mode generation

- `test_rover.py` provides some tests of rover behavior

Notebooks
/////////////////////////////////////////////

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
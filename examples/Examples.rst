.. _fmdtools_examples_repo:

Examples
==============================================

This repository provides several examples models in the `/examples` directory, with some corresponding documentation in the form of notebooks (which may be accessed here). 

Pump
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`examples/ex_pump <pump/readme.rst>` is an example of a simple pump model to demonstrate various capabilities of fmdtools. This includes a tutorial notebook, demostration of plot capabilities, optimization and stochastic modeling.

.. toctree::
   :hidden:
   
   ../examples/pump/readme.rst
   ../examples/pump/Tutorial_complete.ipynb
   ../examples/pump/Pump_Example_Notebook.ipynb
   ../examples/pump/Parallelism_Tutorial.ipynb
   ../examples/pump/Optimization.ipynb
   ../examples/pump/Stochastic_Modelling.ipynb

Multirotor
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/multirotor <../examples/multirotor/readme>` includes several models of drones modelled at differing levels of detail. Includes a demonstration of how models can be matured as more details are added and how the system can be co-optimized.
 
 .. toctree::
   :hidden:

   ../examples/multirotor/readme.rst
   ../examples/multirotor/Demonstration.ipynb
   ../examples/multirotor/Urban_Drone_Demo.ipynb
   ../examples/multirotor/Multirotor_Optimization.ipynb
 
Tank
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/tank <../examples/tank/readme.rst>`_ provides a fairly simple model of a tank, inlet valve, and outlet valve. It includes a demonstration of the model and optimization of said model.

.. toctree::
   :hidden:
   
   ../examples/tank/readme.rst
   ../examples/tank/Tank_Analysis.ipynb
   ../examples/tank/Tank_Optimization.ipynb


EPS
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/eps <../examples/eps/readme.rst>`_ provides a model of a simple electric power system, which shows how undirected propagation can be used in a simple static (i.e., one time-step) moelling use-case.

.. toctree::
   :hidden:
   
   ../examples/eps/readme.rst
   ../examples/eps/EPS_Example_Notebook.ipynb

Action Sequence Graph
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/asg_demo <../examples/asg_demo/readme.rst>`_ provides a very basic model of an Action Architecture, otherwise known as an Action Sequence Graph.

.. toctree::
   :hidden:
   
   ../examples/asg_demo/readme.rst
   ../examples/asg_demo/Action_Sequence_Graph.ipynb

Rover
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/rover <../examples/rover/readme.rst>`_ showcases more advanced methodologies that can be used in fmdtools, and has essentially been the developers’ demo case study for advancing the state-of-the-art in resilience simulation.

.. toctree::
   :hidden:
   
   ../examples/rover/readme.rst
   ../examples/rover/Rover_Setup_Notebook.ipynb
   ../examples/rover/Model_Structure_Visualization_Tutorial.ipynb
   ../examples/rover/FaultSample_Use-Cases.ipynb
   ../examples/rover/ParameterSample_Use-Cases.ipynb
   ../examples/rover/HFAC_Analyses/HFAC_Analyses.ipynb
   ../examples/rover/HFAC_Analyses/IDETC_Human_Paper_Analysis.ipynb
   ../examples/rover/degradation_modelling/Degradation_Modelling_Notebook.ipynb
   ../examples/rover/fault_sampling/Rover_Mode_Notebook.ipynb
   ../examples/rover/optimization/Search_Comparison.ipynb
   ../examples/rover/optimization/Rover_Response_Optimization.ipynb

Multiflow Demo
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/multiflow_demo <../examples/multiflow_demo/readme.rst>`_ provides a is limited to the model in the file `multiflow_demo.py`, which shows basic usage of the :class:`~fmdtools.define.flow.multiflow.MultiFlow` and :class:`~fmdtools.define.flow.commsflow.CommsFlow` flow classes.

.. toctree::
   :hidden:
   
   ../examples/multiflow_demo/readme.rst

Taxiway
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/taxiway <../examples/taxiway/readme.rst>`_ provides a demonstration of how to create a multiagent, systems-of-systems model in fmdtools using the case study of (piloted and unpiloted) aircraft taxiing on a taxiway. 


.. toctree::
   :hidden:
   
   ../examples/taxiway/readme.rst
   ../examples/taxiway/Paper_Notebook.ipynb

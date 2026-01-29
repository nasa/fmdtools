.. _fmdtools_examples_repo:

Examples
==============================================

This repository provides several examples models in the `/examples` directory, with some corresponding documentation in the form of notebooks (which may be accessed here). 

Pump
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`examples/ex_pump <../examples/pump/readme>` is an example of a simple pump model to demonstrate various capabilities of fmdtools. This includes a tutorial notebook, demostration of plot capabilities, optimization and stochastic modeling.

.. toctree::
   :hidden:
   
   ../examples/pump/readme.rst
   ../examples/pump/Tutorial_complete.ipynb
   ../examples/pump/Pump_Example_Notebook.ipynb
   ../examples/pump/Parallelism_Tutorial.ipynb
   ../examples/pump/Optimization.ipynb
   ../examples/pump/Stochastic_Modelling.ipynb

Multirotor Drone
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/multirotor_drone <../examples/multirotor_drone/readme>` includes several models of drones modelled at differing levels of detail. Includes a demonstration of how models can be matured as more details are added and how the system can be co-optimized.
 
 .. toctree::
   :hidden:

   ../examples/multirotor_drone/readme.rst
   ../examples/multirotor_drone/paper_ijphm_fmdtools.ipynb
   ../examples/multirotor_drone/demo_urban_flight.ipynb
   ../examples/multirotor_drone/multirotor_drone_Optimization.ipynb
 
Tank
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/tank <../examples/tank/readme>` provides a fairly simple model of a tank, inlet valve, and outlet valve. It includes a demonstration of the model and optimization of said model.

.. toctree::
   :hidden:
   
   ../examples/tank/readme.rst
   ../examples/tank/Tank_Analysis.ipynb
   ../examples/tank/Tank_Optimization.ipynb


Electric Power System
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/electric_power_system <../examples/electric_power_system/readme>` provides a model of a simple electric power system, which shows how undirected propagation can be used in a simple static (i.e., one time-step) moelling use-case.

.. toctree::
   :hidden:
   
   ../examples/electric_power_system/readme.rst
   ../examples/electric_power_system/Demo_StaticModels.ipynb

Human Hazard Mitigation
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/human_hazard_mitigation <../examples/human_hazard_mitigation/readme>` provides a very basic model of an Action Architecture, otherwise known as an Action Sequence Graph.

.. toctree::
   :hidden:
   
   ../examples/human_hazard_mitigation/readme.rst
   ../examples/human_hazard_mitigation/Tutorial_ActionArchitecture.ipynb

Rover
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/rover <../examples/rover/readme>` showcases more advanced methodologies that can be used in fmdtools, and has essentially been the developers’ demo case study for advancing the state-of-the-art in resilience simulation.

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

State Communication
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/state_communication <../examples/state_communication/readme>` provides a is limited to the model in the file `state_communication.py`, which shows basic usage of the :class:`~fmdtools.define.flow.multiflow.MultiFlow` and :class:`~fmdtools.define.flow.commsflow.CommsFlow` flow classes.

.. toctree::
   :hidden:
   
   ../examples/state_communication/readme.rst
   ../examples/state_communication/Tutorial_MultiFlow_and_CommsFlow.ipynb

Taxiway
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/taxiway <../examples/taxiway/readme>` provides a demonstration of how to create a multiagent, systems-of-systems model in fmdtools using the case study of (piloted and unpiloted) aircraft taxiing on a taxiway. 


.. toctree::
   :hidden:
   
   ../examples/taxiway/readme.rst
   ../examples/taxiway/Paper_Notebook.ipynb


Airspace Library
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/airspacelib <../examples/airspacelib/readme>` provides a demonstration of developing a library to model aircraft in a shared airspace, demonstrating how fmdtools can be used to build new libraries for domain-specific applications.

.. toctree::
   :hidden:
   
   ../examples/airspacelib/readme.rst
   ../examples/airspacelib/wildfireresponse/wildfire_response_demo_presentation.md
   ../examples/airspacelib/wildfireresponse/Wildfire_Demo.ipynb
   ../examples/airspacelib/wildfireresponse/paper_demo.ipynb
   ../examples/airspacelib/contingencymanagement/overview_presentation.md
   ../examples/airspacelib/contingencymanagement/demo_notebook.ipynb
   ../examples/airspacelib/contingencymanagement/proxthreat_notebook.ipynb
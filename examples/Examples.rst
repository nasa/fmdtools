.. _fmdtools_examples_repo:

Examples
==============================================

This repository provides several examples models in the `/examples` directory, with some corresponding documentation in the form of notebooks (which may be accessed here). 

Water Pump
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`examples/model_main <../examples/water_pump/readme>` is an example of a simple pump model to demonstrate various capabilities of fmdtools. This includes a tutorial notebook, demostration of plot capabilities, optimization and stochastic modeling.

.. toctree::
   :hidden:
   
   ../examples/water_pump/readme.rst
   ../examples/water_pump/demo_fault_analysis.ipynb
   ../examples/water_pump/tutorial_fmdtools_basics.ipynb
   ../examples/water_pump/tutorial_optimization.ipynb
   ../examples/water_pump/tutorial_parallelism.ipynb
   ../examples/water_pump/tutorial_stochastic_behavior.ipynb

Multirotor Drone
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/multirotor_drone <../examples/multirotor_drone/readme>` includes several models of drones modelled at differing levels of detail. Includes a demonstration of how models can be matured as more details are added and how the system can be co-optimized.
 
 .. toctree::
   :hidden:

   ../examples/multirotor_drone/readme.rst
   ../examples/multirotor_drone/demo_overview.ipynb
   ../examples/multirotor_drone/tutorial_fmdtools_basics.ipynb
   ../examples/multirotor_drone/paper_ijphm_fmdtools.ipynb
   ../examples/multirotor_drone/demo_urban_flight.ipynb
   ../examples/multirotor_drone/demo_optimization_architectures.ipynb
 
Cooling Tank
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/tank <../examples/cooling_tank/readme>` provides a fairly simple model of a tank, inlet valve, and outlet valve. It includes a demonstration of the model and optimization of said model.

.. toctree::
   :hidden:
   
   ../examples/cooling_tank/readme.rst
   ../examples/cooling_tank/demo_tank_model.ipynb
   ../examples/cooling_tank/paper_jmd_optimization.ipynb


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

Navigating Rover
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/navigating_rover <../examples/navigating_rover/readme>` showcases more advanced methodologies that can be used in fmdtools, and has essentially been the developers’ demo case study for advancing the state-of-the-art in resilience simulation.

.. toctree::
   :hidden:
   
   ../examples/navigating_rover/readme.rst
   ../examples/navigating_rover/demo_rover_model.ipynb
   ../examples/navigating_rover/demo_degradation.ipynb
   ../examples/navigating_rover/demo_response_optimization.ipynb
   ../examples/navigating_rover/tutorial_model_structure_visualization.ipynb
   ../examples/navigating_rover/tutorial_FaultSample.ipynb
   ../examples/navigating_rover/tutorial_ParameterSample.ipynb
   ../examples/navigating_rover/paper_ifac_human.ipynb
   ../examples/navigating_rover/paper_idetc_human.ipynb
   ../examples/navigating_rover/paper_jmd_synthetic_modes.ipynb
   ../examples/navigating_rover/paper_aiaa_coevolution.ipynb

State Communication
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/state_communication <../examples/state_communication/readme>` provides a is limited to the model in the file `state_communication.py`, which shows basic usage of the :class:`~fmdtools.define.flow.multiflow.MultiFlow` and :class:`~fmdtools.define.flow.commsflow.CommsFlow` flow classes.

.. toctree::
   :hidden:
   
   ../examples/state_communication/readme.rst
   ../examples/state_communication/Tutorial_MultiFlow_and_CommsFlow.ipynb

Airport Taxiway
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/airport_taxiway <../examples/airport_taxiway/readme>` provides a demonstration of how to create a multiagent, systems-of-systems model in fmdtools using the case study of (piloted and unpiloted) aircraft taxiing on a taxiway. 


.. toctree::
   :hidden:
   
   ../examples/airport_taxiway/readme.rst
   ../examples/airport_taxiway/paper_jcise_dsa.ipynb


Airspace Library
+++++++++++++++++++++++++++++++++++++++++++++

:doc:`/examples/airspacelib <../examples/airspacelib/readme>` provides a demonstration of developing a library to model aircraft in a shared airspace, demonstrating how fmdtools can be used to build new libraries for domain-specific applications.

.. toctree::
   :hidden:
   
   ../examples/airspacelib/readme.rst
   ../examples/airspacelib/wildfire_response/demo_overview.md
   ../examples/airspacelib/wildfire_response/demo_wildfire.ipynb
   ../examples/airspacelib/wildfire_response/paper_aiaa_optimal_location.ipynb
   ../examples/airspacelib/contingency_management/demo_overview.md
   ../examples/airspacelib/contingency_management/demo_contingency.ipynb
   ../examples/airspacelib/contingency_management/demo_proxthreat.ipynb
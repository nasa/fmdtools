Airspace Resilience Modeling Library
+++++++++++++++++++++++++++++++++++++++++++++

The Airspace Resilience Modeling Library (airspacelib) is a library that can be used to simulate drones and other aircraft interacting with their environment at a high level. The goal of the airspacelib package is to enable the rapid simulation of drones in the airspace. It also provides a demonstration of how one can develop a domain-specific library using fmdtools constructs, such as aircraft in a shared airspace.

The airspacelib package extends the fmdtools modeling and simulation library with drone-specific classes to represent drone behaviors (e.g., 3D trajectory, navigation, etc).

It also provides a number of worked examples of drone modelling in a range of use-cases, including drone in-flight contingency management and wildfire response.

You can view a short overview of each in the :doc:`Wildfire Response Demo Presentation <wildfire_response/demo_overview>`, :doc:`Contingency Management Demo Presentation <contingency_management/demo_overview>`, and :doc:`Water Rescue Demo Presentation <water_rescue/demo_overview>`

Further description and use of the model is given in the provided notebooks:

.. base-gallery::
   :caption: Airspace Resilience Modeling Library
   :tooltip:

   wildfire_response/demo_overview
   wildfire_response/demo_wildfire
   wildfire_response/paper_aiaa_optimal_location
   contingency_management/demo_overview
   contingency_management/demo_contingency
   contingency_management/demo_proxthreat
   water_rescue/demo_overview

**References:**

Hulse, D. E., Mbaye, S., & Davies, M. D. (2025). Determining Optimal Asset Location for Rapid and Efficient Wildfire Suppression: A Simulation-Based Approach. In AIAA SCITECH 2025 Forum (p. 0451).



Package Structure
---------------------------------------------

The airspacelib package has the following structure:

- airspacelib.base: base classes that can be adapted to new case studies

- airspacelib.contingency_management: case study modelling a drone that must perform in-flight contingency management to respond to battery depletion and airspace intrusion scenarios.

- airspacelib.wildfire_response: case study modeling a set of drones in a wildfire response situation. 

- airspacelib.water_rescue: case study of modeling water rescue supported by drones.
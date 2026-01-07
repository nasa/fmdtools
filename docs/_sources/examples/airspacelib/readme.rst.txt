Overview
---------------------------------------------

The Airspace Resilience Modelling Library (airspacelib) is a library that can be used to simulate drones and other aircraft interacting with their environment at a high level. The goal of the airspacelib package is to enable the rapid simulation of drones in the airspace.

The airspacelib package extends the fmdtools modeling and simulation libary with drone-specific classes to represent drone behaviors (e.g., 3D trajectory, navigation, etc).

It also provides a number of worked examples of drone modelling in a range of use-cases, including drone in-flight contingency management and wildfire response.

Structure
/////////////////////////////////////////////


The airspacelib package has the following structure:

- airspacelib.base: base classes that can be adapted to new case studies

- airspacelib.contingencymanagement: case study modelling a drone that must perform in-flight contingency management to respond to battery depletion and airspace intrusion scenarios, including:

  - :doc:`Wildfire Response Demo <wildfire_response_demo_presentation>` provides a high-level overview of the model and analyses

  - :doc:`Wildfire Demo Notebook <Wildfire_Demo>` is a notebook showing some of the basics of modelling setup.

  - :doc:`Paper Demo Notebook <paper_demo>` provides some demonstration of optimizing base placements.

- airspacelib.wildfireresponse: case study modelling a set of drones in a wildfire response situation. The documentation for this module includes:

  - :doc:`Overview presentation <overview_presentation>`, a high-level overview 

  - :doc:`Demo notebook <demo_notebook>`,  a notebook showing some of the basics of modelling setup.

  - :doc:`Proxtheat notebook <proxthreat_notebook>`, a notebook showing how proximity to threat functionality can be evaluated in the model.


References
/////////////////////////////////////////////

- Hulse, D. E., Mbaye, S., & Davies, M. D. (2025). Determining Optimal Asset Location for Rapid and Efficient Wildfire Suppression: A Simulation-Based Approach. In AIAA SCITECH 2025 Forum (p. 0451).

Examples
==============================================

This repository provides several examples models in the `/examples` directory, with some corresponding documentation in the form of notebooks (which may be accessed here). 

Pump
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/ex_pump` is an example of a simple pump model to demonstrate various capabilities of fmdtools. This includes a tutorial notebook, demostration of plot capabilities, optimization and stochastic modeling.

Multirotor
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/multirotor` includes several models of drones modelled at differing levels of detail. Includes a demonstration of how models can be matured as more details are added and how the system can be co-optimized.
 
Tank
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/tank` provides a fairly simple model of a tank, inlet valve, and outlet valve. It includes a demonstration of the model and optimization of said model.

EPS
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/eps` provides a model of a simple electric power system, which shows how undirected propagation can be used in a simple static (i.e., one time-step) moelling use-case.

Action Sequence Graph
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/asg_demo` provides a very basic model of an Action Architecture, otherwise known as an Action Sequence Graph.

Rover
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/rover` showcases more advanced methodologies that can be used in fmdtools, and has essentially been the developers’ demo case study for advancing the state-of-the-art in resilience simulation.

Multiflow Demo
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/multiflow_demo` provides a is limited to the model in the file `multiflow_demo.py`, which shows basic usage of the :class:`fmdtools.define.flow.multiflow.MultiFlow` and :class:`fmdtools.define.flow.commsflow.CommsFlow` flow classes.

Taxiway
+++++++++++++++++++++++++++++++++++++++++++++

`/examples/taxiway` provides a demonstration of how to create a multiagent, systems-of-systems model in fmdtools using the case study of (piloted and unpiloted) aircraft taxiing on a taxiway. 


.. toctree::
   :hidden:
   
   ../examples/pump/readme.rst
   ../examples/multirotor/readme.rst
   ../examples/tank/readme.rst
   ../examples/eps/readme.rst
   ../examples/asg_demo/readme.rst
   ../examples/rover/readme.rst
   ../examples/multiflow_demo/readme.rst
   ../examples/taxiway/readme.rst

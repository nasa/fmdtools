fmdtools.sim 
=========================

The :mod:`fmdtools.sim` package is used to simulate :class:`fmdtools.define.Model` models. It consists of four modules: 
 - :mod:`fmdtools.sim.propagate`, which simulates the user-defined behaviors of a model over set time(s).
 - :mod:`fmdtools.sim.scenario`, which defines scenario information for simulations.
 - :mod:`fmdtools.sim.sample`, which provides classes for defining sets of scenarios to simulate. 
 - :mod:`fmdtools.sim.search`, which provides an the :class:`ProblemInterface` and `DynamicInterface` classes for enabling the search of parameters and/or scenarios.

fmdtools.sim.propagate 
----------------------------------

.. image:: figures/simulation_types.png
   :width: 800

The :mod:`fmdtools.sim.propagate` module is used to simulate the behaviors of a :class:`fmdtools.define.block.Simulable` (`Model` or `Block`) with and without faults. As shown above, each of the methods (described below) fit a given simulation use-case for resilience assessment--single/multiple scenarios, in nominal/faulty scenarios, and at a single set or multiple sets of parameters.

.. automodule:: fmdtools.sim.propagate
   :members:
   :undoc-members:
   :show-inheritance:
   
fmdtools.sim.scenario
----------------------------------

.. automodule:: fmdtools.sim.scenario
   :members:
   :undoc-members:
   :show-inheritance:
   
fmdtools.sim.approach
----------------------------------

.. automodule:: fmdtools.sim.approach
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.sim.search
----------------------------------

.. automodule:: fmdtools.sim.search
   :members:
   :undoc-members:
   :show-inheritance:

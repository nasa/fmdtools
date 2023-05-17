fmdtools.sim 
=========================

The :mod:`fmdtools.sim` package is used to simulate :class:`fmdtools.define.Model` models. It consists of two modules: 

 - :mod:`fmdtools.sim.networks`, which conducts network assessment on a given model's network (which does not require classes for function blocks or behaviors to be defined), and
 
 - :mod:`fmdtools.sim.propagate`, which simulates the user-defined behaviors of a model over set time(s).

fmdtools.sim.networks 
---------------------------------

.. automodule:: fmdtools.sim.networks
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.sim.propagate 
----------------------------------

.. image:: figures/simulation_types.png
   :width: 800

The :mod:`fmdtools.sim.propagate` module is used to simulate the behaviors of a :class:`fmdtools.define.Model` model with and without faults. As shown above, each of the methods (described below) fit a given simulation use-case for resilience assessment--single/multiple scenarios, in nominal/faulty scenarios, and at a single set or multiple sets of parameters.

.. automodule:: fmdtools.sim.propagate
   :members:
   :undoc-members:
   :show-inheritance:


fmdtools.faultsim 
=========================

The :mod:`fmdtools.faultsim` package is used to simulate :class:`fmdtools.modeldef.Model` models. It consists of two modules: 

 - :mod:`fmdtools.faultsim.networks`, which conducts network assessment on a given model's network (which does not require classes for function blocks or behaviors to be defined), and
 
 - :mod:`fmdtools.faultsim.propagate`, which simulates the user-defined behaviors of a model over set time(s).

fmdtools.faultsim.networks 
---------------------------------

.. automodule:: fmdtools.faultsim.networks
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.faultsim.propagate 
----------------------------------

.. image:: figures/simulation_types.png
   :width: 800

The :mod:`fmdtools.faultsim.propagate` module is used to simulate the behaviors of a :class:`fmdtools.modeldef.Model` model with and without faults. As shown above, each of the methods (described below) fit a given simulation use-case for resilience assessment--single/multiple scenarios, in nominal/faulty scenarios, and at a single set or multiple sets of parameters.

.. automodule:: fmdtools.faultsim.propagate
   :members:
   :undoc-members:
   :show-inheritance:


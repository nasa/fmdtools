fmdtools.analyze package
===========================

.. image:: figures/analyze.png
   :width: 800

The analyze package is organized into the  :mod:`fmdtools.analyze.process`, :mod:`fmdtools.analyze.plot`, :mod:`fmdtools.analyze.graph`, and  :mod:`fmdtools.analyze.tabulate` modules, as shown above. :mod:`fmdtools.analyze.process` is used to process simulation results into convenient metrics and statistics for an analysis. The rest of the modules can be thought of as *convenience interfaces* for their respective packages, where:

- :mod:`plot` creates plots in ``matplotlib`` for simulation results (e.g., model histories, end-state classifications, etc).

- :mod:`graph` creates visualizations of the model graph using ``NetworkX``, ``Netgraph`` and/or ``Graphviz`` packages.

- :mod:`tabulate` creates ``pandas`` tables of desired simulation metrics.

The model reference for each of these is provided below:

fmdtools.analyze.graph 
--------------------------------

.. automodule:: fmdtools.analyze.graph
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.plot 
-------------------------------

.. automodule:: fmdtools.analyze.plot
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.process 
----------------------------------

.. automodule:: fmdtools.analyze.process
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.tabulate 
-----------------------------------

.. automodule:: fmdtools.analyze.tabulate
   :members:
   :undoc-members:
   :show-inheritance:


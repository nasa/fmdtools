fmdtools.resultdisp package
===========================

.. image:: figures/resultdisp.png
   :width: 800

The resultdisp package is organized into the  :mod:`fmdtools.resultdisp.process`, :mod:`fmdtools.resultdisp.plot`, :mod:`fmdtools.resultdisp.graph`, and  :mod:`fmdtools.resultdisp.tabulate` modules, as shown above. :mod:`fmdtools.resultdisp.process` is used to process simulation results into convenient metrics and statistics for an analysis. The rest of the modules can be thought of as *convenience interfaces* for their respective packages, where:

- :mod:`plot` creates plots in ``matplotlib`` for simulation results (e.g., model histories, end-state classifications, etc).

- :mod:`graph` creates visualizations of the model graph using ``NetworkX``, ``Netgraph`` and/or ``Graphviz`` packages.

- :mod:`tabulate` creates ``pandas`` tables of desired simulation metrics.

The model reference for each of these is provided below:

fmdtools.resultdisp.graph 
--------------------------------

.. automodule:: fmdtools.resultdisp.graph
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.resultdisp.plot 
-------------------------------

.. automodule:: fmdtools.resultdisp.plot
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.resultdisp.process 
----------------------------------

.. automodule:: fmdtools.resultdisp.process
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.resultdisp.tabulate 
-----------------------------------

.. automodule:: fmdtools.resultdisp.tabulate
   :members:
   :undoc-members:
   :show-inheritance:


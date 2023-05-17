fmdtools.analyze package
===========================

.. image:: figures/analyze.png
   :width: 800

The analyze package is organized into the :mod:`fmdtools.analyze.plot`, :mod:`fmdtools.analyze.graph`, and  :mod:`fmdtools.analyze.tabulate` modules, as shown above. These modules can be of as *convenience interfaces* for their respective packages, where:

- :mod:`plot` creates plots in ``matplotlib`` for simulation results (e.g., model histories, end-state classifications, etc).

- :mod:`graph` creates graph visualizations and analyses of the model structures using ``NetworkX``, and/or ``Graphviz`` packages.

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

fmdtools.analyze.tabulate 
-----------------------------------

.. automodule:: fmdtools.analyze.tabulate
   :members:
   :undoc-members:
   :show-inheritance:


fmdtools.analyze package
===========================

.. image:: figures/analyze_module_structure.png
   :width: 800

The analyze package is organized into the modules:

- :mod:`fmdtools.analyze.result`, which defines the :class:`fmdtools.analyze.result.Result` and :class:`fmdtools.analyze.result.History` classes for tracking, saving, and processing simulation outputs. 
- :mod:`fmdtools.analyze.plot`, which provides functions for plotting `History` and `Result` metrics (and is essentially a convenience interface for `matplotlib`)
- :mod:`fmdtools.analyze.graph`, which provides classes for creating and visualizing Graphs of simulation structures (and is a sort of convenience iterface for `networkx`/`graphviz` and other graphing libraries).
- :mod:`fmdtools.analyze.tabulate`, which provides functions to generate tables of metrics of interest using `pandas` (e.g., FMEAs).
- :mod:`fmdtools.analyze.show`, which shows geometric and other spacial aspects (e.g., shapes, coords, trajectories) of the model using `matplotlib`.

The model reference for each of these is provided below:

fmdtools.analyze.result 
--------------------------------

.. automodule:: fmdtools.analyze.result
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.plot 
-------------------------------

.. automodule:: fmdtools.analyze.plot
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.graph 
--------------------------------

.. automodule:: fmdtools.analyze.graph
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.tabulate 
-----------------------------------

.. automodule:: fmdtools.analyze.tabulate
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.show
-----------------------------------

.. automodule:: fmdtools.analyze.show
   :members:
   :undoc-members:
   :show-inheritance:


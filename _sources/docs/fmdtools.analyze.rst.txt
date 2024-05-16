fmdtools.analyze
===========================
.. automodule:: fmdtools.analyze

.. image:: figures/powerpoint/analyze_module_structure.png
   :width: 800

The analyze package is organized into the modules:

- :mod:`~fmdtools.analyze.common`, which is used for common analysis functions
- :mod:`~fmdtools.analyze.result`, which defines the :class:`~fmdtools.analyze.result.Result` class for sim results
- :mod:`~fmdtools.analyze.history`, which defines the :class:`~fmdtools.analyze.history.History` class for tracking, saving, and processing simulation logs/histories. 
- :mod:`~fmdtools.analyze.graph`, which provides classes for creating and visualizing Graphs of simulation structures (and is a sort of convenience iterface for `networkx`/`graphviz` and other graphing libraries).
- :mod:`~fmdtools.analyze.tabulate`, which provides functions to generate tables and visualizations of metrics of interest using (e.g., FMEAs).
- :mod:`~fmdtools.analyze.phases`, which enables the analysis of phase information from model histories.

The model reference for each of these is provided below:

fmdtools.analyze.common
-------------------------------

.. automodule:: fmdtools.analyze.common
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.result 
--------------------------------

.. automodule:: fmdtools.analyze.result
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.analyze.history
--------------------------------

.. automodule:: fmdtools.analyze.history
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


fmdtools.analyze.phases
-----------------------------------

.. automodule:: fmdtools.analyze.phases
   :members:
   :undoc-members:
   :show-inheritance:
Module Reference
================

.. image:: figures/module_organization.PNG
   :width: 800

The fmdtools package is split into three modules for design, simulation, and analysis, as shown above. The :mod:`fmdtools.modeldef` module provides constructs for model and simulation definition, the :mod:`fmdtools.faultsim` subpackage provides functions to simulate these models and the  the :mod:`fmdtools.resultdisp` subpackage provides functions to analyze and visualize the results of these simulations.

Thus, working with fmdtools often means creating a model file which extends classes from :mod:`fmdtools.modeldef`, and then simulating and analyzing that model in a script or notebook using the :mod:`fmdtools.faultsim` and :mod:`fmdtools.resultdisp` subpackages. This page provides references for the functions and classes in these modules. 

**Submodule Links**

.. toctree::
   :maxdepth: 4

   fmdtools.faultsim
   fmdtools.resultdisp

fmdtools.modeldef
------------------------

.. image:: figures/model_definition.png
   :width: 800

The :mod:`fmdtools.modeldef` module provides constructs for model and simulation definition, as shown above. In general, to define a model, these classes are extended by the user in a model file to define the specific attributes of the functions, flows, components, etc. The module reference is provided below:


.. automodule:: fmdtools.modeldef
   :members:
   :undoc-members:
   :show-inheritance:


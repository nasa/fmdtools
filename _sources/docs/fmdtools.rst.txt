Module Reference
================

.. image:: figures/module_organization.PNG
   :width: 800

The fmdtools package is split into three modules for design, simulation, and analysis, as shown above. The :mod:`fmdtools.define` module provides constructs for model and simulation definition, the :mod:`fmdtools.sim` subpackage provides functions to simulate these models and the  the :mod:`fmdtools.analyze` subpackage provides functions to analyze and visualize the results of these simulations.

Thus, working with fmdtools often means creating a model file which extends classes from :mod:`fmdtools.define`, and then simulating and analyzing that model in a script or notebook using the :mod:`fmdtools.sim` and :mod:`fmdtools.analyze` subpackages. This page provides references for the functions and classes in these modules. 

**Submodule Links**

.. toctree::
   :maxdepth: 4

   fmdtools.sim
   fmdtools.analyze

fmdtools.define
------------------------

.. image:: figures/model_definition.png
   :width: 800

The :mod:`fmdtools.define` module provides constructs for model and simulation definition, as shown above. In general, to define a model, these classes are extended by the user in a model file to define the specific attributes of the functions, flows, components, etc. The module reference is provided below:


.. automodule:: fmdtools.define
   :members:
   :undoc-members:
   :show-inheritance:


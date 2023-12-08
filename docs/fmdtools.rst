Module Reference
================

.. image:: figures/module_organization.svg
   :width: 800

The fmdtools package is split into three subpackages for design, simulation, and analysis. As shown above:

* The :mod:`fmdtools.define` subpackage provides classes for model and simulation definition, and is used for defining system (nominal and faulty) behavior.

* The :mod:`fmdtools.sim` subpackage provides functions to simulate these models in a variety of configurations (e.g., nominal and faulty scenarios). 

* The :mod:`fmdtools.analyze` subpackage provides classes for processing, analyzing, and visualizing simulation results.

Thus, working with fmdtools often means creating a model file which extends classes from :mod:`fmdtools.define`, and then simulating and analyzing that model in a script or notebook using the :mod:`fmdtools.sim` and :mod:`fmdtools.analyze` subpackages. This page provides references for the functions and classes in these modules. 

**Submodule Links**

.. toctree::
   :maxdepth: 5
   
   fmdtools.define
   fmdtools.define.container
   fmdtools.sim
   fmdtools.analyze



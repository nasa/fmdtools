Module Reference
================

.. image:: figures/uml/module_organization.svg
   :width: 800

The fmdtools package is split into three subpackages for design, simulation, and analysis. As shown above. Thus, working with fmdtools often means:

#. Creating a model file which extends classes from :mod:`~fmdtools.define`,
#. Simulating and analyzing that model in a script or notebook using the :mod:`~fmdtools.sim`, and
#. Analyzing simulation results using the :mod:`~fmdtools.analyze` subpackage.

This page provides references for the functions and classes in these packages, which can further be explored below:

.. autosummary::

	fmdtools.define
	fmdtools.sim
	fmdtools.analyze


.. toctree::
   :hidden:
   
   fmdtools.define
   fmdtools.sim
   fmdtools.analyze



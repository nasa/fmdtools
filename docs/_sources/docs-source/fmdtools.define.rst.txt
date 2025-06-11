fmdtools.define
===========================
.. automodule:: fmdtools.define

The define package provides the building blocks to develop a simulation. Simulations are defined in the sub-classes of the :class:`~fmdtools.define.block.base.Simulable` class (in the :mod:`~fmdtools.define.block` and :mod:`~fmdtools.define.architecture` subpackages), as shown below:

.. figure:: figures/uml/block_inheritance.svg
   :width: 800
   :alt: Inheritance of simulable fmdtools classes
   
   Structure of simulable fmdtools subclasses used for developing simulations. 
 
Aside from their internal methods defining behavior, events/indicators, and results, Simulations are additionally composed of internal containers (or sub-attributes) of the class which are defined in their own class.

.. autosummary::

	fmdtools.define.architecture
	fmdtools.define.block
	fmdtools.define.flow
	fmdtools.define.object
	fmdtools.define.base
	fmdtools.define.environment

.. toctree::
   :maxdepth: 1
   :hidden:

   fmdtools.define.architecture
   fmdtools.define.block
   fmdtools.define.container
   fmdtools.define.flow
   fmdtools.define.object


fmdtools.define.base
--------------------------------

.. automodule:: fmdtools.define.base
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.environment
--------------------------------

.. automodule:: fmdtools.define.environment
   :members: Environment
   :show-inheritance:

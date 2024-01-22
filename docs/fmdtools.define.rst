fmdtools.define
===========================

The define package provides the building blocks to develop a simulation. Simulations are defined in the sub-classes of the :class:`Simulable` class (in the :mod:`block` and :mod:`architecture` subpackages), as shown below:

.. figure:: figures/block_inheritance.svg
   :width: 800
   :alt: Structure simulable fmdtools classes
   
   Structure of simulable fmdtools subclasses used for developing simulations. 
 
Aside from their internal methods defining behavior, events/indicators, and results, Simulations are additionally composed of internal containers (or sub-attributes) of the class which are defined in their own class. 

**Submodule Links**

.. toctree::
   :maxdepth: 5

   fmdtools.define.architecture
   fmdtools.define.block
   fmdtools.define.container
   fmdtools.define.flow
   fmdtools.define.object

fmdtools.define.base
--------------------------------

Common methods and data structures are kept in :mod:`base`.

.. automodule:: fmdtools.define.base
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.environment
--------------------------------

.. automodule:: fmdtools.define.environment
   :members: Environment
   :show-inheritance:

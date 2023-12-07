fmdtools.define package
===========================

The define package provides the building blocks to develop a simulation. Simulations are defined in the sub-classes of the :class:`Simulable` class (in the :mod:`block` and :mod:`model` subpackages), as shown below:

.. figure:: figures/block_inheritance.svg
   :width: 800
   :alt: Structure simulable fmdtools classes
   
   Structure of simulable fmdtools subclasses used for developing simulations. 
 
Aside from their internal methods defining behavior, events/indicators, and results, Simulations are additionally composed of internal :term:`role` s, or sub-attributes of the class which are defined in their own class. 

Common methods and data structures are kept in :mod:`common`.


fmdtools.define.block
--------------------------------

.. figure:: figures/fxnblock_structure.png
   :width: 800
   :alt: Structure of a FxnBlock
   
   Code template for :class:`FxnBlock` used to define high-level system functions and their behavior.


.. automodule:: fmdtools.define.block
   :members:
   :undoc-members:
   :show-inheritance:
   
fmdtools.define.common
--------------------------------

.. automodule:: fmdtools.define.common
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.flow
--------------------------------

.. automodule:: fmdtools.define.flow
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.model
--------------------------------

.. figure:: figures/model_structure.png
   :width: 800
   :alt: Structure of a Model
   
   Code template for :class:`Model` used to define the high-level function-flow structure of a system model.

.. automodule:: fmdtools.define.model
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.environment
--------------------------------

.. automodule:: fmdtools.define.environment
   :members: Environment
   :show-inheritance:

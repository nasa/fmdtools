fmdtools.define package
===========================

The define package provides the building blocks to develop a simulation. Simulations are defined in the sub-classes of the :class:`Simulable` class (in the :mod:`block` and :mod:`model` subpackages), as shown below:

.. figure:: figures/block_inheritance.svg
   :width: 800
   :alt: Structure simulable fmdtools classes
   
   Structure of simulable fmdtools subclasses used for developing simulations. 
 
Aside from their internal methods defining behavior, events/indicators, and results, Simulations are additionally composed of internal :term:`role` s, or sub-attributes of the class which are defined in their own class. 

These sub-attributes are provided in the modules:

* :mod:`block`: for :class:`CompArch` and :class:`ASG` aggomerations of :class:`Component` and :class:`Action` blocks, respectively,
* :mod:`state`: for :class:`State`, which represents values of the simulation which change over time,
* :mod:`flow`: for :class:`Flow`, which represents variables/states shared between :class:`Block` s (e.g., in a :class:`Model`),
* :mod:`mode`: for :class:`Mode`, which represents discrete modes (nominal and faulty) which the system may progress through over time,
* :mod:`parameter`: for :class:`Parameter`, which represents variables which do not change over time,
* :mod:`rand`: for :class:`Rand`, which represents random states and behavior, and
* :mod:`time`: for :class:`Time`, which represents the internal time and timers of the block.

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

fmdtools.define.mode
--------------------------------

.. automodule:: fmdtools.define.mode
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

fmdtools.define.parameter
--------------------------------

.. automodule:: fmdtools.define.parameter
   :members:
   :undoc-members:

fmdtools.define.rand
--------------------------------

.. automodule:: fmdtools.define.rand
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.state
--------------------------------

.. automodule:: fmdtools.define.state
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.time
--------------------------------

.. automodule:: fmdtools.define.time
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.geom
--------------------------------

.. automodule:: fmdtools.define.geom
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.environment
--------------------------------

.. automodule:: fmdtools.define.environment
   :members:
   :undoc-members:
   :show-inheritance:


fmdtools.define.coords
--------------------------------

.. automodule:: fmdtools.define.coords
   :members:
   :undoc-members:
   :show-inheritance:
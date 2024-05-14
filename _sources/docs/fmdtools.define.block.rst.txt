fmdtools.define.block
===========================
.. automodule:: fmdtools.define.block

The block subpackage provides a representation of blocks, which are simulable models which may contain containers and other properties.

These variants of block are provided in the following modules:

* :mod:`fmdtools.define.block.base`: for :class:`Simulable`, which is used for all simulations (including architectures) and :class:`Block`, which is used for blocks.
* :mod:`fmdtools.define.block.action`: for :class:`Action`, which is used to represent actions.
* :mod:`fmdtools.define.block.component`: for :class:`Component`, which is used to represent components.
* :mod:`fmdtools.define.block.function`: for :class:`Function`, which is used to represent functions.

fmdtools.define.block.base
--------------------------------

.. automodule:: fmdtools.define.block.base
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.block.action
--------------------------------

.. automodule:: fmdtools.define.block.action
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.block.component
--------------------------------

.. automodule:: fmdtools.define.block.component
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.block.function
--------------------------------
.. automodule:: fmdtools.define.block.function

Functions are used to represent overall system functionality and behaviors (i.e., what a system does).

Functions are defined by extending the Function class, which may then be instantiated, as shown below:

.. figure:: figures/Function.svg
   :width: 800
   :alt: Structure of a Function Class
   
   Example of a Function class and its corresponding instantiation.

To define a function class, it can be helpful to use this code template:

.. figure:: figures/fxnblock_structure.png
   :width: 800
   :alt: Structure of a Function Class
   
   Code template for :class:`Function` used to define high-level system functions and their behavior.


.. autoclass:: fmdtools.define.block.function.Function
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fmdtools.define.block.function.GenericFunction
   :members:
   :undoc-members:
   :show-inheritance:

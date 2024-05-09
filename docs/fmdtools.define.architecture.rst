fmdtools.define.architecture
===========================

The architecture subpackage provides a representation of Architectures, which may be used to represent agglomerations/interactions of blocks in an overall combined simulation.

Different types of architectures are provided in the following modules:

* :mod:`base`: for :class:`Architecture`, which is used as the base class for all architectures.
* :mod:`action`: for :class:`ActionArchitecture`, which is used to represent action sequence graphs
* :mod:`component`: for :class:`ComponentArchitecture`, which is used to represent sets of components.
* :mod:`function`: for :class:`FunctionArchitecture`, which is used to represent functional architectures.
* :mod:`geom`: for :class:`GeomArchitecture`, which is used to represent multiple geometries in an `Environment`.

fmdtools.define.architecture.base
--------------------------------

.. automodule:: fmdtools.define.architecture.base
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.architecture.action
--------------------------------

Class to define successive actions taken by an agent.

For an example and illustration of the structure, see: `The Action Sequence Graph Demo Model <../examples/asg_demo/readme.rst>`_.

.. automodule:: fmdtools.define.architecture.action
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.architecture.component
--------------------------------

.. automodule:: fmdtools.define.architecture.component
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.architecture.function
--------------------------------

   Class used to define the high-level function-flow structure of a system model.

.. figure:: figures/model_structure.png
   :width: 800
   :alt: Structure of a Model

.. automodule:: fmdtools.define.architecture.function
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.architecture.geom
--------------------------------

.. automodule:: fmdtools.define.architecture.geom
   :members:
   :undoc-members:
   :show-inheritance:
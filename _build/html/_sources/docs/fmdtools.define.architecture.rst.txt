fmdtools.define.architecture
===========================
.. automodule:: fmdtools.define.architecture

The architecture subpackage provides a representation of :term:`Architecture` s, which may be used to represent aggregations/interactions of blocks in an overall combined simulation.

Different types of architectures are provided in the following modules:

* :mod:`~fmdtools.define.architecture.base`: for :class:`~fmdtools.define.architecture.base.Architecture`, which is used as the base class for all architectures.
* :mod:`~fmdtools.define.architecture.action`: for :class:`~fmdtools.define.architecture.action.ActionArchitecture`, which is used to represent action sequence graphs
* :mod:`~fmdtools.define.architecture.component`: for :class:`~fmdtools.define.architecture.component.ComponentArchitecture`, which is used to represent sets of components.
* :mod:`~fmdtools.define.architecture.function`: for :class:`~fmdtools.define.architecture.function.FunctionArchitecture`, which is used to represent functional architectures.
* :mod:`~fmdtools.define.architecture.geom`: for :class:`~fmdtools.define.architecture.geom.GeomArchitecture`, which is used to represent multiple geometries in an `Environment`.

fmdtools.define.architecture.base
--------------------------------

.. automodule:: fmdtools.define.architecture.base
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.architecture.action
--------------------------------
.. automodule:: fmdtools.define.architecture.action

The ActionArchitecture class is used to represent a sequenced of action taken by an agent. It is a composition of instantiated Action objects, Flow objects, and conditions as shown below:

.. figure:: ../examples/asg_demo/demo_model_asg.svg
   :width: 800
   :alt: Structure of an ActionArchitecture
   
   Example Action Architecture connecting action objects with flow objects and conditions.

In the Human action architecture, a human operator percieves and acts on a hazard and the total number of hazards acted on is recorded in the outcome flow. This can also be represented in :term:`FRDL` using:

.. figure:: ../examples/asg_demo/demo_model_asg_FRDL.svg
   :width: 600
   :alt: Structure of an ActionArchitecture in FRDL
   
   Propagation network of example Action Architecture.

For more info on this example, see: `The Action Sequence Graph Demo Model <../examples/asg_demo/readme.rst>`_.

.. autoclass:: fmdtools.define.architecture.action.ActionArchitecture
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
.. automodule:: fmdtools.define.architecture.function

The FunctionArchitecture class is used to define the high-level function-flow structure of a system model. It is a composition of instantiated Function and Flow objects, as shown below:

.. figure:: figures/uml/FunctionArchitecture.svg
   :width: 800
   :alt: Structure of a Model
   
   Example Function Architecture connecting function objects with flow objects.

In :term:`FRDL`, this object may be represented as:

.. figure:: figures/uml/FRDL_Function_Example.svg
   :width: 800
   :alt: Example Function Represented in FRDL
   
   Example Function Architecture propagation structure represented in FRDL.

To define a FunctionArchitecture class, it can be helpful to use the following code template:

.. figure:: figures/powerpoint/model_structure.png
   :width: 800
   :alt: Structure of a Model
   
   Code template for FunctionArchitecture classes.

.. autoclass:: fmdtools.define.architecture.function.FunctionArchitecture
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.architecture.geom
--------------------------------

.. automodule:: fmdtools.define.architecture.geom
   :members:
   :undoc-members:
   :show-inheritance:
fmdtools.define.flow
===========================
.. automodule:: fmdtools.define.flow

The flow subpackage provides a representation of flows, which are used to connect Blocks in an Architecture. Different types of flows are provided in the following modules, as shown below/

.. figure:: figures/uml/Flowtypes.svg
   :width: 400
   :alt: Inheritance of flow types classes
   
   Different types of flows defined in fmdtools.

These are provided in the modules:

.. autosummary::

	fmdtools.define.flow.base
	fmdtools.define.flow.multiflow
	fmdtools.define.flow.commsflow


fmdtools.define.flow.base
--------------------------------

.. automodule:: fmdtools.define.flow.base

Flow classes are used to represent variables that are shared between blocks, such as connections or a shared environment. Like blocks, flows (see example below) can hold containers (e.g., States, Parameters, etc.) in order to represent different properties:
 

.. figure:: figures/uml/Flow.svg
   :width: 800
   :alt: example flow class
   
   Example of defining and instantiating a :class:`Flow` class to hold x/y fields.

The following template shows the basic syntax used to define simple flows in a system:

.. figure:: figures/powerpoint/flow_structure.svg
   :width: 600
   :alt: Structure of a Flow Class
   
   Flow class template/example.

.. autoclass:: fmdtools.define.flow.base.Flow
   :members:
   :show-inheritance:

fmdtools.define.flow.multiflow
--------------------------------

.. automodule:: fmdtools.define.flow.multiflow
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.flow.commsflow
--------------------------------

.. automodule:: fmdtools.define.flow.commsflow
   :members:
   :undoc-members:
   :show-inheritance:

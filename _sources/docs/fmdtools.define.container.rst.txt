fmdtools.define.container
===========================
.. automodule:: fmdtools.define.container

The containers subpackage provides the elemental building blocks (i.e., containers for holding states, modes, etc.) for developing simulations, shown below.

.. figure:: figures/uml/Containers.svg
   :width: 800
   :alt: fmdtools container classes
   
   Container classes in fmdtools and their inheritance.

These classes are provided in the following modules:

* :mod:`base`: for :class:`BaseContainer`, which all the other containers inherit from.
* :mod:`mode`: for :class:`Mode`, which represents discrete modes (nominal and faulty) which the system may progress through over time,
* :mod:`state`: for :class:`State`, which represents values of the simulation which change over time,
* :mod:`parameter`: for :class:`Parameter`, which represents variables which do not change over time,
* :mod:`rand`: for :class:`Rand`, which represents random states and behavior, and
* :mod:`time`: for :class:`Time`, which represents the internal time and timers of the block.

fmdtools.define.container.base
--------------------------------

	Base class/module for containers.

.. automodule:: fmdtools.define.container.base
   :members:
   :undoc-members:
   :show-inheritance:


fmdtools.define.container.mode
--------------------------------

.. automodule:: fmdtools.define.container.mode
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.container.state
--------------------------------

State classes are used to represent mutables properties of the system that change over time. State classes are extended and deployed by the user, as shown below: 

.. figure:: figures/uml/State.svg
   :width: 600
   :alt: example state class
   
   Example of extending the :class:`State` class to hold x/y fields.

.. autoclass:: fmdtools.define.container.state.State
   :members: State
   :show-inheritance:


fmdtools.define.container.parameter
--------------------------------

Parameter classes are used to represent immutable properties of the system. Parameter classes are extended and deployed by the user, as shown below: 

.. figure:: figures/uml/Parameter.svg
   :width: 600
   :alt: example state class
   
   Example of extending the :class:`Parameter` class to hold x/y/z fields.

.. autoclass:: fmdtools.define.container.parameter.Parameter
   :members:
   :show-inheritance:

fmdtools.define.container.rand
--------------------------------

.. automodule:: fmdtools.define.container.rand
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.container.time
--------------------------------

.. automodule:: fmdtools.define.container.time
   :members:
   :undoc-members:
   :show-inheritance:


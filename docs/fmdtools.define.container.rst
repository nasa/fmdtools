fmdtools.define.container
===========================
.. automodule:: fmdtools.define.container

The containers subpackage provides the elemental building puzzle pieces needed (i.e., containers for holding states, modes, etc.) to develop simulations, shown below.

.. figure:: figures/uml/Containers.svg
   :width: 800
   :alt: fmdtools container classes
   
   Container classes in fmdtools and their inheritance.

These classes are provided in the following modules:

* :mod:`~fmdtools.define.container.base`: for :class:`~fmdtools.define.container.base.BaseContainer`, which all the other containers inherit from.
* :mod:`~fmdtools.define.container.mode`: for :class:`~fmdtools.define.container.mode.Mode`, which represents discrete modes (nominal and faulty) which the system may progress through over time,
* :mod:`~fmdtools.define.container.state`: for :class:`~fmdtools.define.container.state.State`, which represents values of the simulation which change over time,
* :mod:`~fmdtools.define.container.parameter`: for :class:`~fmdtools.define.container.parameter.Parameter`, which represents variables which do not change over time,
* :mod:`~fmdtools.define.container.rand`: for :class:`~fmdtools.define.container.rand.Rand`, which represents random states and behavior, and
* :mod:`~fmdtools.define.container.time`: for :class:`~fmdtools.define.container.time.Time`, which represents the internal time and timers of the block.

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


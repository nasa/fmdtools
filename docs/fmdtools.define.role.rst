fmdtools.define.container subpackage
===========================

The roles subpackage provides the building blocks for developing simulations like blocks and architectures.

These sub-attributes are provided in the follwoing modules:

* :mod:`mode`: for :class:`Mode`, which represents discrete modes (nominal and faulty) which the system may progress through over time,
* :mod:`state`: for :class:`State`, which represents values of the simulation which change over time,
* :mod:`parameter`: for :class:`Parameter`, which represents variables which do not change over time,
* :mod:`rand`: for :class:`Rand`, which represents random states and behavior, and
* :mod:`time`: for :class:`Time`, which represents the internal time and timers of the block.
* :mod:`geom`: for :class:`Geom`, class and subclasses which represent geometric attributes.
* :mod:`coords`: for :class:`Coords`, which is used to define coordinate systems.

fmdtools.define.container.mode
--------------------------------

.. automodule:: fmdtools.define.container.mode
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.container.state
--------------------------------

.. automodule:: fmdtools.define.container.state
   :members: State
   :show-inheritance:


fmdtools.define.container.parameter
--------------------------------

.. automodule:: fmdtools.define.container.parameter
   :members:
   :undoc-members:

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

fmdtools.define.container.geom
--------------------------------

.. automodule:: fmdtools.define.container.geom
   :members: Geom, GeomPoint, PointParam, GeomLine, LineParam, GeomPoly, PolyParam, GeomArch
   :show-inheritance:


fmdtools.define.container.coords
--------------------------------

.. automodule:: fmdtools.define.container.coords
   :members: Coords, CoordsParam
   :show-inheritance:

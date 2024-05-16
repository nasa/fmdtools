fmdtools.define.object
===========================
.. automodule:: fmdtools.define.object

The object subpackage provides a representation of base-level objects (i.e., timers, geometric points/lines, coordinates) which can be used to construct a simulation.

These different object classes are provided in the following modules:

* :mod:`~fmdtools.define.object.base`: for :class:`~fmdtools.define.object.base.BaseObject`, which is used for the base object class used for both objects and blocks/architectures.
* :mod:`~fmdtools.define.object.geom`: for :class:`~fmdtools.define.object.geom.Geom`, class and subclasses which represent geometric attributes.
* :mod:`~fmdtools.define.object.coords`: for :class:`~fmdtools.define.object.coords.Coords`, which is used to define coordinate systems.
* :mod:`~fmdtools.define.object.timer`: for :class:`~fmdtools.define.object.timer.Timer`, which is used to define timers (used in :class:`~fmdtools.define.container.time.Time` containers).

fmdtools.define.object.base
--------------------------------

.. automodule:: fmdtools.define.object.base
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.object.timer
--------------------------------

.. automodule:: fmdtools.define.object.timer
   :members:
   :undoc-members:
   :show-inheritance:

fmdtools.define.object.geom
--------------------------------

.. automodule:: fmdtools.define.object.geom
   :members: Geom, GeomPoint, PointParam, GeomLine, LineParam, GeomPoly, PolyParam, GeomArch
   :show-inheritance:


fmdtools.define.object.coords
--------------------------------

.. automodule:: fmdtools.define.object.coords
   :members: Coords, CoordsParam
   :show-inheritance:

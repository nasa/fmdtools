#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines geometry classes using the shapely library.

For now, this includes static Geoms, with properties tied to parameters and states
representing allocations.

In the future we hope to include Dynamic Geoms, with properties tied to states

Has classes:

- :class:`Geom`: Base geometry class
- :class:`GeomPoint`: Class defining points.
- :class:`PointParam`: Class defining :class:`GeomPoint` attributes.
- :class:`GeomLine`: Class defining lines.
- :class:`LineParam`: Class defining :class:`GeomLine` attributes.
- :class:`GeomPoly`: Class defining polygons.
- :class:`PolyParam`: Class defining :class:`GeomPoly` attributes.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.object.base import BaseObject
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.analyze.common import setup_plot, consolidate_legend

from shapely import LineString, Point, Polygon
from shapely.ops import nearest_points
from typing import ClassVar
from recordclass import astuple
import numpy as np


class Geom(BaseObject):
    """
    Base class for defining geometries.

    Geometry objects are essentially wrappers around various shapely classes used to
    convey the spacial properties of a model.

    ...

    Roles
    -----
    s : State
        State defining mutable geom properties (e.g., allocation, color, etc)
    p : Param
        Parameter defining immutable geom characteristics (e.g., shapely inputs, buffer)
    """

    container_s = State
    container_p = Parameter
    default_track = ['s']
    all_possible = ['s']
    roledicts = ["buffers"]
    immutable_roles = BaseObject.immutable_roles + ['buffer']

    def __init__(self, *args, s={}, p={}, track='default', **kwargs):
        super().__init__(s=s, p=p, track=[], **kwargs)
        self.shape = self.shapely_class(*self.p.as_args())
        # set buffer shapes
        self.init_role_dict("buffer")
        for buffer, rad in self.buffers.items():
            setattr(self, buffer, self.shape.buffer(rad))
        self.init_track(track=track)

    def at(self, pt, buffername='shape'):
        """
        Determine whether the point x, y is within the buffer 'buffername'.

        Parameters
        ----------
        *pt : tuple
            Locations of x, y, z coordinates.
        buffername : str
            Name of buffer property

        Returns
        -------
        at : bool
            Whether x,y is within the buffer.
        """
        buffer = getattr(self, buffername)
        return buffer.covers(Point(*pt))

    def all_at(self, *pt):
        """
        Find all geom attributes (shape, buffers, etc.) containing the point.

        Parameters
        ----------
        *pt : x,y,z
            Locations of x, y, z coordinates.

        Returns
        -------
        all_at : list
            Buffer/shape containing the pt.

        Examples
        --------
        >>> exp = ExPoint()
        >>> exp.all_at(1.0, 1.0)
        ['shape', 'on']
        >>> exp.all_at(1.0, 0.1)
        ['on']
        >>> exp.all_at(0.0, 0.0)
        []
        """
        buffernames = ['shape', *self.buffers]
        all_at = []
        for bname in buffernames:
            if self.at(pt, bname):
                all_at.append(bname)
        return all_at

    def copy(self, *args, **kwargs):
        """Copy the Geom with given *args and **kwargs."""
        cop = self.__class__(*args, **kwargs)
        cop.s.assign(self.s)
        return cop

    def reset(self):
        """Reset the Geom to initial state."""
        self.s = self.container_s(**self._args_s)

    def return_mutables(self):
        return astuple(self.s)

    def vect_to_shape(self, pt, buffername='shape'):
        """
        Gets the vector (x, y) to a given shape.

        Parameters
        ----------
        pt : tuple/list
            Point to get vector from
        buffername : str
            Name of shape/buffer. Default is 'shape'.

        Examples
        --------
        >>> e = ExPoint()
        >>> e.vect_to_shape((0,0))
        array([[1.],
               [1.]])
        >>> e.vect_to_shape((2,0))
        array([[-1.],
               [ 1.]])
        >>> e.vect_to_shape((1,1))
        array([[0.],
               [0.]])
        """
        buffer = getattr(self, buffername)
        geom_pt = Point(pt)
        geom_c, close_pt = nearest_points(geom_pt, buffer)
        vect_to_shape = np.array(close_pt.xy) - np.array(geom_pt.xy)
        return vect_to_shape

    def vect_at_shape(self, pt, buffername='shape', dist_forward=0.1):
        """
        Get the vector (x, y) at a given shape (e.g., direction of a line at pt).

        Parameters
        ----------
        pt : tuple/lost
            Point closest to shape
        buffername : str
            Name of shape/buffer. Default is 'shape'.
        dist_forward : float
            Distance forward along line segment to project. Give a negative number to
            reverse directions.

        Examples
        --------
        >>> e=ExLine(p={'xys':((0,0),(1,1), (1,0))})
        >>> e.vect_at_shape((0,0))
        array([[0.07071068],
               [0.07071068]])
        >>> e.vect_at_shape((1.5,0.5))
        array([[ 0. ],
               [-0.1]])
        """
        buffer = getattr(self, buffername)
        geom_pt = Point(pt)
        geom_c, close_pt = nearest_points(geom_pt, buffer)
        line_dist = buffer.line_locate_point(geom_pt)
        next_pt = buffer.line_interpolate_point(line_dist + dist_forward)
        vect_at_shape = np.array(next_pt.xy) - np.array(close_pt.xy)
        return vect_at_shape

    def show(self, shapes={'all': {}}, fig=None, ax=None, figsize=(4, 4), z=False,
             geomlabel='', **kwargs):
        """
        Show a Geom (shape and buffers) as lines on a plot.

        Parameters
        ----------
        shapes : dict, optional
            Aspects of the Geom to plot and their corresponding plot kwargs.
            The default is {'all': {}}.
        fig : matplotlib.figure, optional
            Existing Figure. The default is None.
        ax : matplotlib.axis, optional
            Existing axis. The default is None.
        figsize : tuple, optional
            Size for figure (if instantiating). The default is (4, 4).
        z : bool/number, optional
            If plotting on a 3d axis, set z to a number which will be the z-level.
            The default is False.
        geomlabel : str, optional
            Overall label for the geom (if desired). The default is ''.
        **kwargs : kwargs
            overall kwargs for plt.plot for all shapes.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        if not ax:
            fig, ax = setup_plot(z=z, figsize=figsize)
        if 'all' in shapes:
            shapes = {'shape': {}, **{v: {} for v in self.buffers}}
        if type(z) in (int, float):
            plot_kwargs = {'zs': z, 'zdir': 'z', **kwargs}
        else:
            plot_kwargs = kwargs
        for shape, shape_kwargs in shapes.items():
            if geomlabel:
                shape_label = geomlabel + "." + shape
            else:
                shape_label = shape
            local_kwargs = {**plot_kwargs, 'label': shape_label, **shape_kwargs}
            shap = getattr(self, shape)
            if isinstance(shap, Point):
                ax.scatter(shap.x, shap.y, **local_kwargs)
            elif isinstance(shap, LineString):
                linecoords = np.array([*shap.coords])
                ax.plot(linecoords[:, 0], linecoords[:, 1], **local_kwargs)
            elif isinstance(shap, Polygon):
                ax.plot(*shap.exterior.xy, **local_kwargs)
        ax.axis('equal')
        consolidate_legend(ax, **kwargs)
        return fig, ax


class PointParam(Parameter):
    """
    Parameter defining points. May be extended with buffers.

    Fields
    ------
    x : float
        x-location of point.
    y : float
        y-location of point.

    Examples
    --------
    a point with center at 1.0, 1.0 and radius of 1.0 defining whether something
    is on the point would be defined by the parameter:

    >>> class ExPointParam(PointParam):
    ...    x: float = 1.0
    ...    y: float = 1.0
    ...    buffer_on: float = 1.0

    >>> expa = ExPointParam()
    >>> expa.as_args()
    ([1.0, 1.0],)
    """

    x: ClassVar[float] = 0.0
    y: ClassVar[float] = 0.0

    def as_args(self):
        """Create arguments for shapely Point class based on fields."""
        return ([self[i] for i in ['x', 'y', 'z'] if i in dir(self)], )


class GeomPoint(Geom):
    """
    Point geometry representing x-y coordinate and buffers.

    Defined by parameter (x, y and buffer(s)) as well as states (properties of
    of point).

    Examples
    --------
    >>> class ExPoint(GeomPoint):
    ...    container_p = ExPointParam
    ...    container_s = ExGeomState
    >>> exp = ExPoint()
    >>> exp.at((1.0, 1.0), "on")
    True
    >>> exp.at((0.0, 0.0), "on")
    False
    >>> exp.at((1.0, 0.1), "on")
    True

    Additionally, note the underlying shapely attribute at .shape:

    >>> type(exp.shape)
    <class 'shapely.geometry.point.Point'>

    as well as the buffer (on):

    >>> type(exp.on)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    container_p = PointParam
    shapely_class = Point


class ExGeomState(State):
    """
    Example Geom state for testing.

    Has 'occupied' property defining whether something is at the geom or not.
    """

    occupied: bool = False


class ExPointParam(PointParam):
    """
    Example point parameter for testing.

    Has a default center at (1.0, 1.0) and radius defining "on" of 1.0.
    """

    x: float = 1.0
    y: float = 1.0
    buffer_on: float = 1.0


class ExPoint(GeomPoint):
    """Example point for testing."""

    container_p = ExPointParam
    container_s = ExGeomState


class LineParam(Parameter):
    """
    Parameter defining lines. May be extended with buffers.

    ...

    Fields
    ------
    xys : tuple
        tuple of points ((x1, y1), ...) defining shapely.LineString.

    Examples
    --------
    A point with ends at (0.0, 0.0) and (1.0, 1.0) and radius of 1.0 defining
    whether something is on the line would be defined by the parameter:

    >>> class ExLineParam(LineParam):
    ...     xys: tuple = ((0.0, 0.0), (1.0, 1.0))
    ...     buffer_on: float = 1.0
    >>> exlp = ExLineParam()
    >>> exlp.as_args()
    (((0.0, 0.0), (1.0, 1.0)),)
    """

    xys: ClassVar[tuple] = ()

    def as_args(self):
        """Create arguments for shapely LineString class based on fields."""
        return (self.xys, )


class ExLineParam(LineParam):
    """Example parameter defining a line with a given buffer 'on'."""

    xys: tuple = ((0.0, 0.0), (1.0, 1.0))
    buffer_on: float = 1.0


class GeomLine(Geom):
    """Point geometry representing a line and possible buffers.

    Defined by parameter (xys and buffer(s)) as well as states (properties of
    of point).

    Examples
    --------
    >>> class ExLine(GeomLine):
    ...    container_p = ExLineParam
    ...    container_s = ExGeomState
    >>> exp = ExLine()
    >>> exp.at((1.0, 1.0), "on")
    True
    >>> exp.at((0.0, 0.0), "on")
    True
    >>> exp.at((2.0, 2.0), "on")
    False

    Additionally, note the underlying shapely attribute at .shape:

    >>> type(exp.shape)
    <class 'shapely.geometry.linestring.LineString'>

    As well as the buffer (on):

    >>> type(exp.on)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    container_p = LineParam
    shapely_class = LineString


class ExLine(GeomLine):
    """Example GeomLine to use in testing."""

    container_p = ExLineParam
    container_s = ExGeomState


class PolyParam(Parameter):
    """
    Parameter defining polygons. May be extended with buffers.

    ...
    
    Fields
    ------
    shell : tuple
        tuple of points ((x1, y1), ...) defining outer shell.
    holes : tuple
        tuple of points defining holes.

    Examples
    --------
    The following PolyParam defines a hollow right triangle:

    >>> class ExPolyParam(PolyParam):
    ...     shell: tuple = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    ...     holes: tuple = (((0.3, 0.2), (0.6, 0.2), (0.6, 0.5)), )
    >>> expp = ExPolyParam()
    >>> expp.as_args()
    (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), (((0.3, 0.2), (0.6, 0.2), (0.6, 0.5)),))
    """

    shell: ClassVar[tuple] = ()
    holes: ClassVar[tuple] = ()

    def as_args(self):
        """Create arguments for shapely Polygon class based on fields."""
        return self.shell, self.holes


class ExPolyParam(PolyParam):
    """Example polygon parameter defining a basic triangle for use in testing."""

    shell: tuple = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    holes: tuple = (((0.3, 0.2), (0.6, 0.2), (0.6, 0.5)), )


class GeomPoly(Geom):
    """
    Polygon geometry defining shape and possible buffers.

    Defined by PolyParam, which is used to instantiate the Polygon class. Also may
    contain a state for the given status (e.g., occupied, red/blue, etc.).

    Examples
    --------
    >>> class ExPoly(GeomPoly):
    ...    container_p = ExPolyParam
    ...    container_s = ExGeomState
    >>> egp = ExPoly()
    >>> egp.at((0.1, 0.05))
    True
    >>> egp.at((0.4, 0.3))
    False

    Additionally, note the underlying shapely attribute at .shape:

    >>> type(egp.shape)
    <class 'shapely.geometry.polygon.Polygon'>
    """

    container_p = PolyParam
    shapely_class = Polygon


class ExPoly(GeomPoly):
    """Example Polygon for use in testing."""

    container_p = ExPolyParam
    container_s = ExGeomState


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    exp = ExPoint()

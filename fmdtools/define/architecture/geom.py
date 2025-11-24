#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines :class:`GeomArchitecture` class for representing multiple geometries in a
:class:`Environment`.

Classes
-------
:class:`GeomArchitecture`: Architecture of multiple geometries.
:class:`ExGeomArch`: Example GeomArchitecture.

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

from fmdtools.define.architecture.base import Architecture
from fmdtools.define.container.parameter import Parameter
from fmdtools.analyze.common import setup_plot
from fmdtools.define.object.geom import GeomPoint, GeomLine, GeomPoly
from fmdtools.define.object.geom import ExPoint, ExLine, ExPoly
from fmdtools.define.block.base import Block


class GeomArchitecture(Architecture):
    """
    Agglomeration of multiple geoms/shapes.

    Architecture is defined using add_shape method in user-defined init_shapes method.

    Examples
    --------
    for an architecture with the geoms already defined:

    >>> class ExGeomArch(GeomArchitecture):
    ...    def init_architecture(self):
    ...        self.add_point("ex_point", ExPoint)
    ...        self.add_line("ex_line", ExLine)
    ...        self.add_poly("ex_poly", ExPoly)
    ...    def dynamic_behavior(self):
    ...        if self.t.time >= 1.0:
    ...            self.points["ex_point"].s.buffer_around = self.t.time
    ...    def static_behavior(self):
    ...        self.lines["ex_line"].s.buffer_around = self.points["ex_point"].s.buffer_around

    This can then be used in containing classes (e.g., environments) that need multiple
    geoms. We can then access the individual geoms in the geoms dict, e.g.,:

    >>> ega = ExGeomArch()
    >>> ega
    exgeomarch ExGeomArch
    - t=Time(time=-0.1, timers={})
    POINTS:
    - ex_point=ExPoint(s=(occupied=False, buffer_around=1.0))
    LINES:
    - ex_line=ExLine(s=(occupied=False, buffer_around=1.0))
    POLYS:
    - ex_poly=ExPoly(s=(occupied=False, buffer_around=1.0))
    >>> ega.geoms()['ex_point'].s
    ExGeomState(occupied=False, buffer_around=1.0)
    >>> ega.h
    time:                         array(101)
    points.ex_point.s.occupied:   array(101)
    points.ex_point.s.buffer_around: array(101)
    lines.ex_line.s.occupied:     array(101)
    lines.ex_line.s.buffer_around: array(101)
    polys.ex_poly.s.occupied:     array(101)
    polys.ex_poly.s.buffer_around: array(101)
    >>> ega.return_mutables()
    ((False, 1.0), (False, 1.0), (False, 1.0), (np.float64(-0.1), np.int32(0), False, False, False))

    GeomArchitectures are also simulable provided dynamic_behavior and static_behavior
    methods as shown below. Note that this behavior must be called externally,
    such as from a function, to have meaning in a broader modelling context.

    >>> ega(time=2.0)
    >>> ega
    exgeomarch ExGeomArch
    - t=Time(time=2.0, timers={})
    POINTS:
    - ex_point=ExPoint(s=(occupied=False, buffer_around=2.0))
    LINES:
    - ex_line=ExLine(s=(occupied=False, buffer_around=2.0))
    POLYS:
    - ex_poly=ExPoly(s=(occupied=False, buffer_around=1.0))
    """

    container_p = Parameter
    __slots__= ("points", "lines", "polys")
    default_track = ['points', 'lines', 'polys']
    all_possible = ['points', 'lines', 'polys']
    flexible_roles = ['point', 'line', 'poly']
    rolename = 'ga'

    def base_type(self):
        """Return fmdtools type of the model class."""
        return GeomArchitecture

    def init_architecture(self, **kwargs):
        """Use this placeholder method to define custom architectures."""
        pass

    def geoms(self):
        """Return a dict of all points, lines, and polygons."""
        return {**self.points, **self.lines, **self.polys}

    def check_geom_class(self, gclass, baseclass):
        """Check that geom class/object inherits from given base class."""
        try:
            if not issubclass(gclass, baseclass):
                raise Exception("gclass "+gclass+" not a "+baseclass.__name__)
        except TypeError:
            if not isinstance(gclass, baseclass):
                raise Exception("gclass "+gclass+" not a "+baseclass.__name__)

    def add_point(self, name, pclass=GeomPoint, **kwargs):
        """
        Add/instantiate an individual point to the overall architecture.

        Parameters
        ----------
        name : str
            Name of the geom object to instantiate.
        gclass : Geom
            Class defining the geom.
        **kwargs : kwargs
            kwargs defining the object for gclass.
        """
        self.check_geom_class(pclass, GeomPoint)
        self.add_flex_role_obj('points', name, objclass=pclass, **kwargs)

    def add_line(self, name, lclass=GeomLine, **kwargs):
        """
        Add/instantiate an individual line to the overall architecture.

        Parameters
        ----------
        name : str
            Name of the geom object to instantiate.
        gclass : Geom
            Class defining the geom.
        **kwargs : kwargs
            kwargs defining the object for gclass.
        """
        self.check_geom_class(lclass, GeomLine)
        self.add_flex_role_obj('lines', name, objclass=lclass, **kwargs)

    def add_poly(self, name, pclass=GeomPoly, **kwargs):
        """
        Add/instantiate an individual polygon to the overall architecture.

        Parameters
        ----------
        name : str
            Name of the geom object to instantiate.
        gclass : Geom
            Class defining the geom.
        **kwargs : kwargs
            kwargs defining the object for gclass.
        """
        self.check_geom_class(pclass, GeomPoly)
        self.add_flex_role_obj('polys', name, objclass=pclass, **kwargs)

    def all_at(self, *pt):
        """
        Find all geoms (and buffers) a given is at.

        Parameters
        ----------
        *pt : x,y
            x, y, z location to check.

        Returns
        -------
        all_at : dict
            Names of geoms where the point is at (and their properties)

        Examples
        --------
        >>> exga = ExGeomArch()
        >>> exga.all_at(1.0, 1.0)
        {'ex_point': ['shape', 'on', 'around'], 'ex_line': ['shape', 'on', 'around'], 'ex_poly': ['shape', 'around']}
        >>> exga.all_at(0.0, 0.0)
        {'ex_line': ['shape', 'on', 'around'], 'ex_poly': ['shape', 'around']}
        >>> exga.all_at(0.4, 0.3)
        {'ex_point': ['on', 'around'], 'ex_line': ['on', 'around'], 'ex_poly': ['around']}
        """
        all_at = {}
        for geomname, geom in self.geoms().items():
            at_geom = geom.all_at(*pt)
            if at_geom:
                all_at[geomname] = at_geom
        return all_at

    def show(self, geoms={'all': {}}, fig=None, ax=None, figsize=(4, 4), z=False,
             **kwargs):
        """
        Show the shapes of a GeomArchitecture all on one plot.

        Parameters
        ----------
        geoms : dict, optional
            Individual shapes to plot and their corresponding kwargs.
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
        **kwargs : kwargs
            Overall kwargs to show.geom for all geoms.

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        if not ax:
            fig, ax = setup_plot(z=z, figsize=figsize)
        if 'all' in geoms:
            geoms = {g: {'shapes': 'all'} for g in self.geoms()}

        for geomname, geom_kwargs in geoms.items():
            local_kwargs = {**kwargs, 'geomlabel': geomname, **geom_kwargs}
            fig, ax = self.geoms()[geomname].show(ax=ax, fig=fig, z=z, **local_kwargs)
        return fig, ax

    def show_from(self, hist, t, **kwargs):
        """
        Show the GeomArch at the given time in the provided history.

        Parameters
        ----------
        hist : History
            History of states for the GeomArchitecture.
        t : int
            Time in the history to show the state of the GeomArchitecture at.
        **kwargs : kwargs
            Keyword arguments to GeomArchitecture.show().

        Returns
        -------
        fig : figure
            Matplotlib figure object
        ax : axis
            Corresponding matplotlib axis
        """
        for flex_role in self.flexible_roles:
            for geomname, geom in self.get_flex_role_objs(flex_role).items():
                geom.assign_from(hist.get(flex_role+"s."+geomname), t)
        return self.show(**kwargs)

    def prop_static(self):
        """Since geoms are not connected, just run in sequence."""
        Block.update_arch_behaviors(self, "static")

    def build(self, construct_graph=False, **kwargs):
        """Build the action graph."""
        super().build(construct_graph=construct_graph, **kwargs)


class ExGeomArch(GeomArchitecture):
    """Example Geometric Architecture for testing etc."""

    def init_architecture(self, **kwargs):
        """Initialize example geoms."""
        self.add_point("ex_point", ExPoint)
        self.add_line("ex_line", ExLine)
        self.add_poly("ex_poly", ExPoly)

    def dynamic_behavior(self):
        """Example dynamic behavior demonstrating dynamic buffers."""
        if self.t.time >= 0.0:
            self.points["ex_point"].s.buffer_around = self.t.time

    def static_behavior(self):
        self.lines["ex_line"].s.buffer_around = self.points["ex_point"].s.buffer_around


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    ega = ExGeomArch()

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

    This can then be used in containing classes (e.g., environments) that need multiple
    geoms. We can then access the individual geoms in the geoms dict, e.g.,:

    >>> ega = ExGeomArch()
    >>> ega.geoms()['ex_point'].s
    ExGeomState(occupied=False)
    >>> ega.h
    points.ex_point.s.occupied:   array(101)
    lines.ex_line.s.occupied:     array(101)
    polys.ex_poly.s.occupied:     array(101)
    >>> ega.return_mutables()
    ((False,), (False,), (False,), (-0.1, 0, 0.0, 1))
    """

    container_p = Parameter
    default_track = ['points', 'lines', 'polys']
    all_possible = ['points', 'lines', 'polys']
    flexible_roles = ['points', 'lines', 'polys']
    rolename = 'ga'

    def base_type(self):
        """Return fmdtools type of the model class."""
        return GeomArchitecture

    def init_architecture(self, **kwargs):
        """Use this placeholder method to define custom architectures."""
        a = 1

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
        {'ex_point': ['shape', 'on'], 'ex_line': ['shape', 'on'], 'ex_poly': ['shape']}
        >>> exga.all_at(0.0, 0.0)
        {'ex_line': ['shape', 'on'], 'ex_poly': ['shape']}
        >>> exga.all_at(0.4, 0.3)
        {'ex_point': ['on'], 'ex_line': ['on']}
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


class ExGeomArch(GeomArchitecture):
    """Example Geometric Architecture for testing etc."""

    def init_architecture(self, **kwargs):
        """Initialize example geoms."""
        self.add_point("ex_point", ExPoint)
        self.add_line("ex_line", ExLine)
        self.add_poly("ex_poly", ExPoly)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    ega = ExGeomArch()

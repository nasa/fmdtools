# -*- coding: utf-8 -*-
"""
Now:
    Static Geoms, with properties tied to parameters and states representing
    allocations.
Future:
    Dynamic Geoms, with properties tied to states
"""
from fmdtools.define.object.base import BaseObject
from fmdtools.define.container.parameter import Parameter
from fmdtools.analyze.common import get_sub_include
from fmdtools.analyze.history import History, init_indicator_hist
from fmdtools.analyze.common import setup_plot
from fmdtools.define.object.geom import Geom, GeomPoint, GeomLine, GeomPoly
from fmdtools.define.object.geom import ExPoint, ExLine, ExPoly
from recordclass import asdict


class GeomArchitecture(BaseObject):
    """
    Agglomeration of multiple geoms/shapes.

    Architecture is defined using add_shape method in user-defined init_shapes method.

    Examples
    --------
    for an architecture with the geoms already defined:
    >>> class ExGeomArch(GeomArchitecture):
    ...    def init_geoms(self):
    ...        self.add_geom("ex_point", ExPoint)
    ...        self.add_geom("ex_line", ExLine)
    ...        self.add_geom("ex_poly", ExPoly)

    This can then be used in containing classes (e.g., environments) that need multiple
    geoms. We can then access the individual geoms in the geoms dict, e.g.:

    >>> ega = ExGeomArch()
    >>> ega.geoms['ex_point'].s
    ExGeomState(occupied=False)
    """

    container_p = Parameter
    default_track = ['geoms']
    all_possible = ['geoms']

    def __init__(self, *args, p={}, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.points = []
        self.lines = []
        self.polys = []
        self.geoms = {}
        super().__init__(p=p)
        self.init_geoms(**kwargs)

    def check_role(self, rolename):
        if rolename != 'ga':
            raise Exception("Invalid rolename for GeomArchitecture: "+rolename)

    def init_geoms(self, **kwargs):
        """Use this placeholder method to define custom architectures."""
        a = 1

    def add_geom(self, name, gclass, *args, **kwargs):
        """
        Add/instantiate an individual geom to the overall architecture.

        Parameters
        ----------
        name : str
            Name of the geom object to instantiate.
        gclass : Geom
            Class defining the geom.
        *args : args
            args defining the object for gclass.
        **kwargs : kwargs
            kwargs defining the object for gclass.
        """
        setattr(self, name, gclass(*args, **kwargs))
        if issubclass(gclass, GeomPoint):
            self.points.append(name)
        elif issubclass(gclass, GeomLine):
            self.lines.append(name)
        elif issubclass(gclass, GeomPoly):
            self.polys.append(name)
        elif not issubclass(gclass, Geom):
            raise Exception(name + " gclass " + str(gclass) + " not a Geom")
        self.geoms[name] = getattr(self, name)

    def copy(self):
        """Copy geoms in the architecture (mirrors current states)."""
        cop = self.__class__()
        for geom in self.geoms:
            cop.geoms[geom].s.assign(self.geoms[geom].s)
        return cop

    def reset(self):
        for geom in self.geoms:
            self.geoms[geom].reset()

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
        for geomname, geom in self.geoms.items():
            at_geom = geom.all_at(*pt)
            if at_geom:
                all_at[geomname] = at_geom
        return all_at

    def create_hist(self, timerange, track):
        """
        Create history for the architecture.

        Examples
        --------
        >>> ega = ExGeomArch()
        >>> h = ega.create_hist([0.0], 'default')
        >>> h.flatten()
        geoms.ex_point.s.occupied:      array(1)
        geoms.ex_line.s.occupied:       array(1)
        geoms.ex_poly.s.occupied:       array(1)
        """
        track = self.get_track(track, all_possible=self.all_possible)
        hist = History()
        init_indicator_hist(self, hist, timerange, track)
        geoms_track = get_sub_include('geoms', track)
        if geoms_track:
            hist['geoms'] = History()
            for geomname, geom in self.geoms.items():
                sh = geom.create_hist(timerange,
                                      get_sub_include(geomname, geoms_track))
                if sh:
                    hist.geoms[geomname] = sh
        return hist

    def return_states(self):
        states = {}
        for geomname, geom in self.geoms.items():
            states[geomname] = asdict(geom.s)
        return states

    def return_mutables(self):
        """
        Return all mutables (geom states).

        Examples
        --------
        >>> ega = ExGeomArch()
        >>> ega.return_mutables()
        (False, False, False)
        """
        mutes = []
        for geom in self.geoms.values():
            mutes.extend(geom.return_mutables())
        return tuple(mutes)

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
            geoms = {g: {'shapes': 'all'} for g in self.geoms}

        for geomname, geom_kwargs in geoms.items():
            local_kwargs = {**kwargs, 'geomlabel': geomname, **geom_kwargs}
            fig, ax = self.geoms[geomname].show(ax=ax, fig=fig, z=z, **local_kwargs)
        return fig, ax


class ExGeomArch(GeomArchitecture):
    """Example Geometric Architecture for testing etc."""

    def init_geoms(self):
        """Initialize example geoms."""
        self.add_geom("ex_point", ExPoint)
        self.add_geom("ex_line", ExLine)
        self.add_geom("ex_poly", ExPoly)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    ega = ExGeomArch()

# -*- coding: utf-8 -*-
"""
Classes for creating environments.
"""

from fmdtools.define.parameter import Parameter
from fmdtools.define.rand import Rand
from fmdtools.define.common import is_iter, get_obj_track, init_obj_attr
from fmdtools.analyze.result import History, get_sub_include
from fmdtools.define.flow import CommsFlow
from fmdtools.define.coords import Coords, ExampleCoords
from fmdtools.define.geom import GeomArch, ExGeomArch


class Environment(CommsFlow):
    """
    Class for representing environments (in development).

    Environments are CommsFlows in order to readily enable perception as well as
    sending and recieving of information. In addition to having normal flow properties,
    they also contain the roles:

    TODO: Properly expand create_local, update, send, recieve, etc to use ga and coords.

    Roles
    ---------------
    c: Coords
        Representation of gridworld properties
    r: Rand
        Representaiton of random variables/rng
    ga: GeomArch
        Representaion of shapes/forms
    """

    slots = ["c", "_args_c", "r", "_args_r", "ga", "args_ga"]
    _init_c = Coords
    _init_r = Rand
    _init_ga = GeomArch
    default_track = ('s', 'i', 'c', 'ga')
    all_possible = ('s', 'i', 'c', 'r', 'ga')

    def __init__(self, name, glob=[], p={}, s={}, r={}, c={}, ga={}):
        super().__init__(name, glob=glob, p=p, s=s)
        if 'p' not in c:
            c = {**c, 'p': self.p}
        init_obj_attr(self, r=r, c=c, ga=ga)
        self.update_seed()

    def get_typename(self):
        return "Environment"

    def return_mutables(self):
        return (*super().return_mutables(),
                self.r.return_mutables(),
                self.c.return_mutables(),
                self.ga.return_mutables())

    def copy(self, glob=[], p={}, s={}):
        cop = super().copy(glob=glob, p=p, s=s)
        cop.r.assign(self.r)
        cop.c = self.c.copy()
        cop.ga = self.ga.copy()
        return cop

    def status(self):
        stat = super().status()
        stat["c"] = self.c.return_states()
        stat["ga"] = self.ga.return_states()
        return stat

    def reset(self):
        super().reset()
        self.r.reset()
        self.c = self._init_c(**self._args_c)
        self.ga.reset()

    def update_seed(self, seed=[]):
        if not seed:
            seed = self.r.seed
        self.c.r.update_seed(seed)

    def return_probdens(self):
        return self.r.return_probdens() * self.c.r.return_probdens()

    def create_hist(self, timerange, track):
        CommsFlow.create_hist(self, timerange, track)
        track = get_obj_track(self, track, all_possible=self.all_possible)
        track = [t for t in track if t not in ('s', 'i')]
        for att in track:
            val = getattr(self, att)
            val_h = val.create_hist(timerange, get_sub_include(att, track))
            if val_h:
                self.h[att] = val_h
        return self.h
        


class ExampleEnvironment(Environment):
    _init_c = ExampleCoords
    _init_ga = ExGeomArch


if __name__ == "__main__":
    e = ExampleEnvironment("env")
    d = e.copy()
    import doctest
    doctest.testmod(verbose=True)

# -*- coding: utf-8 -*-
"""
Classes for creating environments.
"""

from fmdtools.define.parameter import Parameter
from fmdtools.define.rand import Rand
from fmdtools.define.common import is_iter, get_obj_track, init_obj_attr
from fmdtools.analyze.result import History
from fmdtools.define.flow import CommsFlow
from fmdtools.define.coords import Coords, ExampleCoords


class Environment(CommsFlow):
    """
    Class for representing environments (in development).

    Environments are CommsFlows in order to readily enable perception as well as
    sending and recieving of information. In addition to having normal flow properties,
    they also contain the roles:

    Roles
    ---------------
    c: Coords
        Representation of gridworld properties
    r: Rand
        Representaiton of random variables/rng
    ga: GeomArch
        (in development): Representaion of shapes/forms
    """

    slots = ["c", "_args_c", "r", "_args_r"]
    _init_c = Coords
    _init_r = Rand
    default_track = ('s', 'i', 'c')

    def __init__(self, name, glob=[], p={}, s={}, c={}, r={}):
        super().__init__(name, glob=glob, p=p, s=s)
        if 'p' not in c:
            c = {**c, 'p': self.p}
        init_obj_attr(self, r=r, c=c)
        self.update_seed()

    def return_mutables(self):
        return (*super().return_mutables(),
                self.r.return_mutables(),
                self.c.return_mutables())

    def copy(self, glob=[], p={}, s={}):
        cop = super().copy(glob=glob, p=p, s=s)
        cop.r.assign(self.r)
        cop.c = self.c.copy()
        return cop

    def status(self):
        stat = super().status()
        stat["c"] = self.c.return_states()
        return stat

    def reset(self):
        super().reset()
        self.r.reset()
        self.c = self._init_c(**self._args_c)

    def update_seed(self, seed=[]):
        if not seed:
            seed = self.r.seed
        self.c.r.update_seed(seed)

    def return_probdens(self):
        return self.r.return_probdens() * self.c.r.return_probdens()

class ExampleEnvironment(Environment):
    _init_c = ExampleCoords


if __name__ == "__main__":
    e = ExampleEnvironment("env")
    d = e.copy()
    import doctest
    doctest.testmod(verbose=True)

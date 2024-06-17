# -*- coding: utf-8 -*-
"""Module for representing environments with the :class:`Environment` class."""
from fmdtools.define.container.rand import Rand
from fmdtools.define.flow.commsflow import CommsFlow
from fmdtools.define.object.coords import Coords, ExampleCoords
from fmdtools.define.architecture.geom import GeomArchitecture, ExGeomArch


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
        Representation of random variables/rng
    ga: GeomArch
        Representaion of shapes/forms

    Examples
    --------
    >>> class ExampleEnvironment(Environment):
    ...    coords_c = ExampleCoords
    ...    arch_ga = ExGeomArch
    >>> env = ExampleEnvironment('env')
    >>> env.create_hist([1.0])
    c.r.probdens:                   array(1)
    c.st:                           array(1)
    ga.points.ex_point:             array(1)
    ga.lines.ex_line:               array(1)
    ga.polys.ex_poly:               array(1)
    """

    slots = ["c", "r", "ga"]
    container_r = Rand
    coords_c = Coords
    arch_ga = GeomArchitecture
    roletypes = ['container', 'coords', 'arch']
    default_track = ('s', 'i', 'c', 'ga')
    all_possible = ('s', 'i', 'c', 'r', 'ga')

    def __init__(self, name='', glob=[], p={}, s={}, r={}, c={}, ga={},
                 track='default'):
        if 'p' not in c and self.coords_c.container_p == self.container_p:
            c = {**c, 'p': p}
        if 'p' not in ga and self.arch_ga.container_p == self.container_p:
            ga = {**ga, 'p': p}
        super().__init__(name=name, glob=glob, p=p, s=s, track=track)
        # NOTE: p and s also init here because if not, they are overritten
        # may need to change in the future
        self.init_roletypes('container', "coords", "arch", r=r, p=p, s=s, c=c, ga=ga)
        self.update_seed()

    def get_typename(self):
        return "Environment"

    def copy(self, glob=[], p={}, s={}):
        """
        Copy the Environment.

        Examples
        --------
        Copies should be identical but independent after copying, e.g.:

        >>> e = ExampleEnvironment("env")
        >>> e.ga.points['ex_point'].s.occupied = True
        >>> e.c.st[0, 0] = 1

        Given these changes, the copy should have the same states (and not default):

        >>> d = e.copy()
        >>> d.ga.points['ex_point'].s.occupied
        True
        >>> d.c.st[0, 0]
        1.0

        It should also be independent, meaning changes don't effect the original:

        >>> d.c.st[0, 1] = 1.0
        >>> e.c.st[0, 1]
        0.0
        >>> d.ga.lines['ex_line'].s.occupied = True
        >>> e.ga.lines['ex_line'].s.occupied
        False
        """
        cop = super().copy(glob=glob, p=p, s=s)
        cop.r.assign(self.r)
        cop.c = self.c.copy()
        cop.ga = self.ga.copy()
        if hasattr(self, 'h'):
            cop.h = self.h.copy()
        return cop

    def status(self):
        stat = super().status()
        stat["c"] = self.c.return_states()
        stat["ga"] = self.ga.return_states()
        return stat

    def reset(self):
        super().reset()
        self.r.reset()
        self.c = self.coords_c(**self._args_c)
        self.ga.reset()

    def update_seed(self, seed=[]):
        if not seed:
            seed = self.r.seed
        self.c.r.update_seed(seed)

    def return_probdens(self):
        return self.r.return_probdens() * self.c.r.return_probdens()


class ExampleEnvironment(Environment):
    """Example environment for testing."""

    coords_c = ExampleCoords
    arch_ga = ExGeomArch


if __name__ == "__main__":
    e = ExampleEnvironment("env")
    d = e.copy()
    import doctest
    doctest.testmod(verbose=True)

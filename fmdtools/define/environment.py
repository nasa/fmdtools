#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module for representing environments with the :class:`Environment` class.

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

    def __init__(self, name='', root='', glob=[], p={}, s={}, r={}, c={}, ga={},
                 track='default'):
        super().__init__(name=name, root=root, glob=glob, p=p, s=s, track=track)
        # NOTE: p and s also init here because if not, they are overritten
        # may need to change in the future
        self.init_roletypes('container', "coords", "arch", r=r, p=p, s=s)
        if 'p' not in c and getattr(self.coords_c, 'container_p', None) == getattr(self, 'container_p', None):
            c = {**c, 'p': p}
        if 'p' not in ga and getattr(self.arch_ga, 'container_p', None) == getattr(self, 'container_p', None):
            ga = {**ga, 'p': p}
        r_kwargs = {'run_stochastic': self.r.run_stochastic, 'seed': self.r.seed}
        c = {**{'r': r_kwargs}, **c}
        ga = {**{'r': r_kwargs}, **ga}
        self.init_roletypes('coords', 'arch', c=c, ga=ga)

    def base_type(self):
        """Return fmdtools type of the model class."""
        return Environment

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

    def reset(self):
        super().reset()
        self.r.reset()
        self.c = self.coords_c(**self._args_c)
        self.ga.reset()

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

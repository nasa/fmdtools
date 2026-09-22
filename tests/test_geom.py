#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing some basic path planner functionality.

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


import pytest
from shapely import LineString, Polygon

from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.object.geom import (
    ExLine, ExPoint, ExPolyCoords, GeomLine, GeomParameter, GeomPoly, PolyParam,
)


SHELL = ((0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0))
HOLES = (((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)),)


class ShellParam(PolyParam):
    """Polygon with a shell, using PolyParam's default empty holes."""
    shell: tuple = SHELL
    buffer_around: float = 0.5


class HoleParam(GeomParameter):
    """Use the parameter base class from issue #36."""
    shell: tuple = SHELL
    holes: tuple = HOLES
    buffer_around: float = 0.5


class MovingPolyState(State):
    """Mutable constructor fields override their parameter counterparts."""
    shell: tuple = SHELL
    holes: tuple = HOLES


class LineParam(GeomParameter):
    """Line coordinate sequence with single-level tuple."""
    coordinates: tuple = ((0.0, 0.0), (2.0, 2.0))
    buffer_on: float = 0.25


class MovingLineState(State):
    """Mutable coordinates override the parameter coordinates."""
    coordinates: tuple = ((3.0, 0.0), (3.0, 2.0))


class ShellGeom(GeomPoly):
    container_p = ShellParam


class HoleGeom(GeomPoly):
    container_p = HoleParam


class MovingPolyGeom(HoleGeom):
    container_s = MovingPolyState


class LineGeom(GeomLine):
    container_p = LineParam


class MovingLineGeom(LineGeom):
    container_s = MovingLineState


def test_polygon_shell_and_default_holes():
    geom = ShellGeom()
    assert geom.get_shape().equals(Polygon(SHELL))
    assert geom.at((3.0, 3.0))
    assert geom.at((6.25, 3.0), 'around')
    assert not geom.at((7.0, 3.0), 'around')


def test_polygon_holes_and_architecture_queries():
    geom = HoleGeom()
    assert geom.get_shape().equals(Polygon(SHELL, HOLES))
    assert geom.at((1.0, 1.0))
    assert not geom.at((3.0, 3.0))
    arch = GeomArchitecture()
    arch.add_geom('zone', HoleGeom)
    assert arch.all_at(1.0, 1.0)['zone'] == ['shape', 'around']
    assert arch.all_at(3.0, 3.0) == {}


def test_polygon_reads_updated_state_fields():
    geom = MovingPolyGeom()
    assert not geom.at((3.0, 3.0))
    geom.s.holes = ()
    assert geom.at((3.0, 3.0))
    geom.s.shell = tuple((x + 10.0, y) for x, y in SHELL)
    assert not geom.at((3.0, 3.0))
    assert geom.at((13.0, 3.0))
    assert geom.at((16.25, 3.0), 'around')
    assert geom.p.shell == SHELL
    assert geom.p.holes == HOLES


@pytest.mark.parametrize('coordinates', [
    ((0.0, 0.0), (2.0, 2.0)),
    ((0.0, 0.0), (1.0, 2.0), (3.0, 3.0)),
    ((0.0, 0.0, 1.0), (2.0, 2.0, 3.0)),
    (),
])
def test_line_accepts_natural_coordinates(coordinates):
    geom = LineGeom(p={'coordinates': coordinates})
    assert geom.get_shape().equals_exact(LineString(coordinates), 0.0)


def test_line_buffers_and_state_overrides():
    geom = MovingLineGeom()
    assert geom.at((3.0, 1.0))
    assert not geom.at((1.0, 1.0))
    geom.s.coordinates = ((5.0, 0.0), (5.0, 2.0))
    assert not geom.at((3.0, 1.0))
    assert geom.at((5.1, 1.0), 'on')
    assert geom.p.coordinates == LineParam().coordinates


def test_empty_line():
    assert ExLine(p={'coordinates': ((),)}).get_shape().is_empty

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


# ============================================================================
# 1. IMPORTS
# ============================================================================
from fmdtools.define.container.state import State
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.object.geom import GeomParameter, GeomPoly, GeomPoint
from fmdtools.define.pathplan import PathPlannerParameter, PathPlannerBase

## TODO: remove reliance on @property definitions


# ============================================================================
# 2. SHARED PARAMETERS
# ============================================================================
class BasePlannerParameter(PathPlannerParameter):
    start_point: tuple = (10.0, 10.0)
    end_point: tuple = (100.0, 100.0)


# ============================================================================
# 3. SHARED PLANNER SETUP UTILITY
# ============================================================================
def setup_planner(arch, environment):
    arch.add_fxn('planner', PathPlannerBase, p={
        'start_point': arch.p.start_point,
        'end_point': arch.p.end_point,
        'blocksize': arch.p.blocksize,
        'agent_radius': arch.p.agent_radius,
        'max_distance': arch.p.max_distance,
        'path_validation_enabled': arch.p.path_validation_enabled,
        'replanning_enabled': arch.p.replanning_enabled,
        'max_replan_attempts': arch.p.max_replan_attempts,
        'collision_check_resolution': arch.p.collision_check_resolution
    })
    arch.fxns['planner'].init_environment(environment)


# ============================================================================
# 4. SHARED STATES
# ============================================================================
class ObstacleState(State):
    cost: float = 1.0
    traversable: bool = False
    goal_allowed: bool = False

class AgentState(State):
    cost: float = 0.0
    traversable: bool = True
    goal_allowed: bool = True


# ============================================================================
# 5. GEOM DEFINITIONS (params + geoms)
# ============================================================================
# --- Obstacle geoms ---
class RestrictedZoneParam(GeomParameter):
    shell: tuple = ((10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0))
    holes: tuple = (((15.0, 15.0), (35.0, 15.0), (35.0, 35.0), (15.0, 35.0)),)

class RestrictedZoneGeom(GeomPoly):
    container_p = RestrictedZoneParam
    container_s = ObstacleState
    def get_shapely_args(self):     # override due to fmdtools bug
        return (self.p.shell, self.p.holes)

class OccupiedPointParam(GeomParameter):
    coordinates: tuple = (60.0, 60.0)
    buffer_around: float = 3.0

class OccupiedPointGeom(GeomPoint):
    container_p = OccupiedPointParam
    container_s = ObstacleState

class SecondPointParam(GeomParameter):
    coordinates: tuple = (70.0, 20.0)
    buffer_around: float = 4.0

class SecondPointGeom(GeomPoint):
    container_p = SecondPointParam
    container_s = ObstacleState

# --- Agent geom ---
class AgentCircleParam(GeomParameter):
    coordinates: tuple = (0.0, 0.0)
    buffer_around: float = 2.5

class AgentCircleGeom(GeomPoint):
    container_p = AgentCircleParam
    container_s = AgentState


# ============================================================================
# 6. GRID ENVIRONMENT
# ============================================================================
class GridEnvironmentParam(CoordsParam):
    """Grid environment parameters."""
    blocksize: float = 10.0

class GridEnvironment(Coords):
    container_p = GridEnvironmentParam
    __slots__ = ('agent_geom',)

    feature_traversable = (bool, True)
    feature_goal_allowed = (bool, True)
    feature_cost = (float, 1.0)

    def init_properties(self, **kwargs):
        self.set_range("traversable", False, xmin=20, xmax=40, ymin=20, ymax=40, inclusive=True)
        self.set_range("goal_allowed", False, xmin=20, xmax=40, ymin=20, ymax=40, inclusive=True)
        self.set_range("cost", 100.0, xmin=20, xmax=40, ymin=20, ymax=40, inclusive=True)

        occupied_cells = [(60, 60), (80, 80)]
        self.set_pts(occupied_cells, "traversable", False)
        self.set_pts(occupied_cells, "goal_allowed", False)
        self.set_pts(occupied_cells, "cost", float('inf'))


# ============================================================================
# 7. HYBRID ENVIRONMENT WRAPPER
# ============================================================================
class HybridEnvironment:
    def __init__(self, grid_env, geom_arch):
        self.grid = grid_env
        self.geom_arch = geom_arch
    def to_index(self, x, y):
        return self.grid.to_index(x, y)
    @property
    def grid_property(self):
        return self.grid.grid


# ============================================================================
# 8. GEOM ARCHITECTURES
# ============================================================================
class ObstacleGeomArch(GeomArchitecture):
    container_p = BasePlannerParameter
    def init_architecture(self, **kwargs):
        self.add_geom("restricted_zone", RestrictedZoneGeom, s={'cost': 100.0, 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_60_60", OccupiedPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_70_20", SecondPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})

class CircleAgentGeomArch(GeomArchitecture):
    container_p = AgentCircleParam
    def init_architecture(self, **kwargs):
        self.add_geom("agent", AgentCircleGeom, p={'buffer_around': self.p.buffer_around}, s={'traversable': True, 'goal_allowed': True})

class CircleAgentObstacleGeomArch(GeomArchitecture):
    container_p = AgentCircleParam
    def init_architecture(self, **kwargs):
        self.add_geom("agent", AgentCircleGeom, p={'buffer_around': self.p.buffer_around}, s={'traversable': True, 'goal_allowed': True})
        self.add_geom("restricted_zone", RestrictedZoneGeom, s={'cost': 100.0, 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_60_60", OccupiedPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_70_20", SecondPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})


# ============================================================================
# 9. CIRCLE AGENT PARAMETER
# ============================================================================
class CircleAgentParameter(PathPlannerParameter):
    start_point: tuple = (10.0, 10.0)
    end_point: tuple = (90.0, 90.0)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "blocksize", 10.0)
        object.__setattr__(self, "agent_radius", 2.5)


# ============================================================================
# 10. FUNCTION ARCHITECTURES
# ============================================================================
# --- Point-agent architectures ---
class GridArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    __slots__ = ('env', 'planner')
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.blocksize) + 1
        y_size = int(self.p.end_point[1] / self.p.blocksize) + 1
        self.env = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.blocksize))
        setup_planner(self, self.env)
        self.planner = self.fxns['planner']

class GeomEnvironmentArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    default_sp = {'end_time': 15}
    __slots__ = ('env', 'planner')
    def init_architecture(self, **kwargs):
        self.env = ObstacleGeomArch()
        setup_planner(self, self.env)
        self.planner = self.fxns['planner']

class HybridArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    default_sp = {'end_time': 15}
    __slots__ = ('grid', 'geom_arch', 'env', 'planner')
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.blocksize) + 1
        y_size = int(self.p.end_point[1] / self.p.blocksize) + 1
        self.grid = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.blocksize))
        self.geom_arch = ObstacleGeomArch()
        self.env = HybridEnvironment(self.grid, self.geom_arch)
        setup_planner(self, self.env)
        self.planner = self.fxns['planner']

# --- Circle-agent architectures ---
class CircleGridArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    __slots__ = ('grid', 'geom_arch', 'env', 'planner',)
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.blocksize) + 1
        y_size = int(self.p.end_point[1] / self.p.blocksize) + 1
        self.grid = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.blocksize))
        self.geom_arch = CircleAgentGeomArch()
        self.env = HybridEnvironment(self.grid, self.geom_arch)
        setup_planner(self, self.env)
        self.planner = self.fxns['planner']

class CircleGeomArchitecture(FunctionArchitecture):
    container_p = CircleAgentParameter
    __slots__ = ('env', 'planner',)
    def init_architecture(self, **kwargs):
        self.env = CircleAgentObstacleGeomArch()
        setup_planner(self, self.env)
        self.planner = self.fxns['planner']

class CircleHybridArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    __slots__ = ('grid', 'geom_arch', 'env', 'planner',)
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.blocksize) + 1
        y_size = int(self.p.end_point[1] / self.p.blocksize) + 1
        self.grid = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.blocksize))
        self.geom_arch = CircleAgentObstacleGeomArch()
        self.env = HybridEnvironment(self.grid, self.geom_arch)
        setup_planner(self, self.env)
        self.planner = self.fxns['planner']


# ============================================================================
# 11. TEST-MODEL FACTORY FUNCTIONS
# ============================================================================
def create_grid_test_model(start=(10.0, 10.0), end=(100.0, 100.0), blocksize=10.0):
    """Create a standard grid-based test model."""
    return GridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'blocksize': blocksize,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })

def create_geom_test_model(start=(10.0, 10.0), end=(100.0, 100.0)):
    """Create a geometry-based test model."""
    return GeomEnvironmentArchitecture(p={
        'start_point': start,
        'end_point': end,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })

def create_hybrid_test_model(start=(10.0, 10.0), end=(100.0, 100.0), blocksize=10.0):
    """Create a hybrid (grid + geometry) test model."""
    return HybridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'blocksize': blocksize,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })

def create_circle_grid_test_model(start=(10.0, 10.0), end=(90.0, 90.0), agent_radius=2.5, blocksize=10.0):
    """Create a circular agent grid-based test model."""
    return CircleGridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'blocksize': blocksize,
        'agent_radius': agent_radius,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })

def create_circle_geom_test_model(start=(10.0, 10.0), end=(90.0, 90.0), agent_radius=2.5):
    """Create a circular agent geometry-based test model."""
    return CircleGeomArchitecture(p={
        'start_point': start,
        'end_point': end,
        'agent_radius': agent_radius,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })

def create_circle_hybrid_test_model(start=(10.0, 10.0), end=(90.0, 90.0), agent_radius=2.5, blocksize=10.0):
    """Create a circular agent hybrid (grid + geometry) test model."""
    return CircleHybridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'blocksize': blocksize,
        'agent_radius': agent_radius,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })



#####################################################################################################
import unittest

class TestGridEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_grid_test_model()

    def test_collision(self):
        test_cases = [((5.0, 5.0), True, "Free space point"),
                    ((15.0, 15.0), False, "Point in restricted zone"),
                    ((25.0, 25.0), False, "Point in restricted zone center"),
                    ((60.0, 60.0), False, "Occupied point"),
                    ((80.0, 80.0), False, "Occupied point"),
                    ((50.0, 50.0), True, "Free space point near boundary"),
                    ((90.0, 90.0), True, "Free space point far from obstacles"),]
        for (x, y), expected, description in test_cases:
            result = self.model.planner.check_collision(x, y)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_goal_feasibility(self):
        test_cases = [((90.0, 90.0), True, "Valid goal in free space"),
                    ((50.0, 50.0), True, "Valid goal away from obstacles"),
                    ((25.0, 25.0), False, "Invalid goal in restricted zone"),
                    ((60.0, 60.0), False, "Invalid goal on occupied cell"),
                    ((82.0, 78.0), False, "Invalid goal on occupied cell"),
                    ((5.0, 5.0), True, "Valid goal at edge of free space"),
                    ((100.0, 100.0), True, "Valid goal at far corner"),]
        for goal, expected, description in test_cases:
            result = self.model.planner.check_goal_feasible(goal)["feasible"]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_segment_collision(self):
        test_cases = [((10.0, 10.0), (14.5, 14.5), True, "Clear segment in free space"),
                    ((5.0, 5.0), (10.0, 10.0), True, "Short clear segment"),
                    ((25.0, 25.0), (35.0, 35.0), False, "Segment through restricted zone"),
                    ((60.0, 60.0), (65.0, 65.0), False, "Segment from occupied cell"),
                    ((50.0, 50.0), (60.0, 60.0), False, "Segment to occupied cell"),
                    ((70.0, 70.0), (74.5, 74.5), True, "Clear segment away from obstacles"),
                    ((19.0, 19.0), (41.0, 41.0), False, "Long segment crossing restricted zone"),
                    ((60.0, 50.0), (60.0, 60.0), False, "Vertical segment hitting occupied"),]
        for start, end, expected, description in test_cases:
            x1, y1 = start
            x2, y2 = end
            result, _ = self.model.planner.check_segment_collision(x1, y1, x2, y2)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_cost_computation(self):
        test_cases = [([(60.0, 10.0), (70.0, 20.0), (80.0, 30.0)], True, "Simple 3-point path in free space"),
                    ([(5.0, 5.0), (50.0, 50.0), (90.0, 90.0)], False, "Path crossing restricted zone"),  
                    ([(10.0, 10.0), (14.0, 14.0)], True, "Short 2-point path in free space"),
                    ([(50.0, 50.0), (60.0, 60.0), (70.0, 70.0)], False, "Path through occupied cell"),
                    ([(25.0, 5.0), (45.0, 5.0), (45.0, 50.0), (90.0, 90.0)], False, "L-shaped path crossing obstacles"),]
        for path, expected, description in test_cases:
            result = self.model.planner.compute_path_cost(path)
            is_valid = (result != float("inf"))
            with self.subTest(msg=description):
                self.assertEqual(is_valid, expected, description)

    def test_path_validation(self):
        test_cases = [([(60.0, 10.0), (70.0, 20.0), (80.0, 30.0)], True, "Valid 3-point path in free space"),
                    ([(5.0, 5.0), (10.0, 10.0)], True, "Valid 2-point path"),
                    ([], False, "Empty path"),
                    ([(10.0, 10.0)], False, "Single point (path too short)"),
                    ([(10.0, 10.0), (25.0, 25.0)], False, "Path through restricted zone"),
                    ([(50.0, 50.0), (60.0, 60.0)], False, "Path through occupied cell"),
                    ([(25.0, 5.0), (30.0, 5.0), (40.0, 50.0)], False, "L-shaped path crossing obstacles"),
                    ([(72.0, 72.0), (82.0, 82.0), (92.0, 92.0)], False, "Multi-part path through occupied cell"),]
        for path, expected, description in test_cases:
            result = self.model.planner.validate_path(path)[0]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

class TestGeomEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_geom_test_model()

    def test_collision(self):
        test_cases = [((5.0, 5.0), True, "Free space point"),
                    ((12.0, 12.0), False, "Point in restricted zone"),
                    ((15.0, 15.0), False, "Point on edge of restricted zone"),
                    ((25.0, 25.0), True, "Point in restricted zone center"),
                    ((60.0, 60.0), False, "Point on occupied obstacle"),
                    ((62.0, 61.0), False, "Point in occupied obstacle"),
                    ((70.0, 20.0), False, "Point on second obstacle"),
                    ((50.0, 50.0), True, "Free space point away from obstacles"),
                    ((75.0, 75.0), True, "Free space point far from obstacles"),]
        for (x, y), expected, description in test_cases:
            result = self.model.planner.check_collision(x, y)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_goal_feasibility(self):
        test_cases = [((75.0, 75.0), True, "Valid goal in free space"),
                    ((50.0, 50.0), True, "Valid goal away from obstacles"),
                    ((12.0, 12.0), False, "Invalid goal in restricted zone"),
                    ((60.0, 60.0), False, "Invalid goal on occupied point"),
                    ((70.0, 20.0), False, "Invalid goal on second obstacle"),
                    ((5.0, 5.0), True, "Valid goal at edge of free space"),]
        for goal, expected, description in test_cases:
            result = self.model.planner.check_goal_feasible(goal)["feasible"]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_segment_collision(self):
        test_cases = [((5.0, 5.0), (9.9, 9.9), True, "Clear segment in free space"),
                    ((50.0, 50.0), (57.0, 57.0), True, "Short clear segment"),
                    ((20.0, 20.0), (35.0, 35.0), False, "Segment through restricted zone"),
                    ((58.0, 58.0), (62.0, 62.0), False, "Segment near occupied point"),
                    ((5.0, 5.0), (75.0, 75.0), False, "Long segment crossing obstacles"),
                    ((65.0, 65.0), (75.0, 75.0), True, "Clear segment away from all obstacles"),]
        for start, end, expected, description in test_cases:
            x1, y1 = start
            x2, y2 = end
            result, _ = self.model.planner.check_segment_collision(x1, y1, x2, y2)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_cost_computation(self):
        test_cases = [([(60.0, 90.0), (70.0, 80.0), (80.0, 70.0)], True, "Simple 3-point path in free space"),
                    ([(5.0, 5.0), (25.0, 25.0), (75.0, 75.0)], False, "Path crossing restricted zone"),
                    ([(50.0, 50.0), (55.0, 55.0)], True, "Short 2-point path in free space"),
                    ([(58.0, 58.0), (62.0, 62.0), (68.0, 68.0)], False, "Path through occupied point"),]
        for path, expected, description in test_cases:
            result = self.model.planner.compute_path_cost(path)
            is_valid = (result != float("inf"))
            with self.subTest(msg=description):
                self.assertEqual(is_valid, expected, description)

    def test_path_validation(self):
        test_cases = [([(60.0, 0.0), (70.0, 10.0), (80.0, 20.0)], True, "Valid 3-point path in free space"),
                    ([(50.0, 50.0), (55.0, 55.0)], True, "Valid 2-point path"),
                    ([], False, "Empty path"),
                    ([(10.0, 10.0)], False, "Single point (path too short)"),
                    ([(10.0, 10.0), (30.0, 30.0)], False, "Path through restricted zone"),
                    ([(58.0, 58.0), (62.0, 62.0)], False, "Path through occupied point"),
                    ([(5.0, 5.0), (25.0, 25.0), (75.0, 75.0)], False, "Path crossing obstacles"),
                    ([(65.0, 65.0), (70.0, 70.0), (75.0, 75.0)], True, "Valid path away from all obstacles"),]
        for path, expected, description in test_cases:
            result = self.model.planner.validate_path(path)[0]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

class TestHybridEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_hybrid_test_model()

    def test_collision(self):
        test_cases = [((5.0, 5.0), True, "Free space point"),
                    ((11.0, 11.0), False, "Point in grid restricted zone"),
                    ((25.0, 25.0), False, "Point in restricted zone hole but coords obstacle"),
                    ((72.0, 19.0), False, "Point in geom obstacle buffer"),
                    ((70.0, 20.0), False, "Point on geom obstacle"),
                    ((63.0, 63.0), False, "Point in coords obstacle"),
                    ((80.0, 20.0), True, "Free space far from obstacles"),
                    ((50.0, 50.0), True, "Free space between obstacles"),]
        for (x, y), expected, description in test_cases:
            result = self.model.planner.check_collision(x, y)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_goal_feasibility(self):
        test_cases = [((80.0, 40.0), True, "Valid goal in free space"),
                    ((50.0, 50.0), True, "Valid goal between obstacles"),
                    ((12.0, 12.0), False, "Invalid goal in grid restricted zone"),
                    ((60.0, 60.0), False, "Invalid goal on geom obstacle"),]
        for goal, expected, description in test_cases:
            result = self.model.planner.check_goal_feasible(goal)["feasible"]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_segment_collision(self):
        test_cases = [((65.0, 25.0), (90.0, 40.0), True, "Clear segment in free space"),
                    ((20.0, 20.0), (35.0, 35.0), False, "Segment through grid restricted zone"),
                    ((60.0, 10.0), (80.0, 30.0), False, "Segment through geom obstacle"),
                    ((75.0, 75.0), (85.0, 85.0), False, "Segment through coords obstacles"),
                    ((5.0, 5.0), (85.0, 85.0), False, "Long segment crossing multiple obstacles"),]
        for start, end, expected, description in test_cases:
            x1, y1 = start
            x2, y2 = end
            result, _ = self.model.planner.check_segment_collision(x1, y1, x2, y2)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_cost_computation(self):
        test_cases = [([(60.0, 90.0), (70.0, 80.0), (80.0, 70.0)], True, "Simple 3-point path in free space"),
                        ([(5.0, 5.0), (25.0, 25.0), (75.0, 75.0)], False, "Path crossing multiple obstacles"),
                        ([(50.0, 30.0), (60.0, 40.0)], True, "Short 2-point path in free space"),
                        ([(60.0, 10.0), (80.0, 30.0)], False, "Path through geom obstacle"),]
        for path, expected, description in test_cases:
            result = self.model.planner.compute_path_cost(path)
            is_valid = (result != float("inf"))
            with self.subTest(msg=description):
                self.assertEqual(is_valid, expected, description)

    def test_path_validation(self):
        test_cases = [([(60.0, 90.0), (70.0, 80.0), (80.0, 70.0)], True, "Valid 3-point path in free space"),
                    ([(50.0, 30.0), (60.0, 40.0)], True, "Valid 2-point path"),
                    ([], False, "Empty path"),
                    ([(10.0, 10.0)], False, "Single point (path too short)"),
                    ([(10.0, 10.0), (30.0, 30.0)], False, "Path through grid restricted zone"),
                    ([(58.0, 58.0), (62.0, 62.0)], False, "Path through geom obstacle"),
                    ([(5.0, 5.0), (85.0, 85.0)], False, "Path crossing multiple obstacles"),]
        for path, expected, description in test_cases:
            result = self.model.planner.validate_path(path)[0]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)


class TestShapeAgentGridEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_circle_grid_test_model()

    def test_collision(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [((70.0, 10.0), True, "Free space - circle center safely in free space"),
                    ((25.0, 41.0), False, "Circle overlaps with restricted zone boundary"),     ## BUG: This one is failing and I don't know why
                    ((30.0, 30.0), False, "Circle center in restricted zone"),
                    ((50.0, 50.0), True, "Circle center away from all obstacles"),
                    ((60.0, 60.0), False, "Circle overlaps with occupied point"),
                    ((70.0, 70.0), True, "Circle clear of obstacles"),
                    ((80.0, 80.0), False, "Circle at occupied point"),
                    ((10.0, 10.0), True, "Circle at start area"),]
        for (x, y), expected, description in test_cases:
            result = self.model.planner.check_collision(x, y, geom=agent_geom)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_goal_feasibility(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [((70.0, 70.0), True, "Goal in free space away from obstacles"),
                    ((50.0, 50.0), True, "Goal at boundary of free space"),
                    ((30.0, 30.0), False, "Goal in restricted zone"),
                    ((60.0, 60.0), False, "Goal at occupied point"),
                    ((80.0, 80.0), False, "Goal at occupied cell"),
                    ((12.0, 12.0), True, "Goal at start area"),
                    ((25.0, 25.0), False, "Goal in obstacle region"),]
        for goal, expected, description in test_cases:
            result = self.model.planner.check_goal_feasible(goal, geom=agent_geom)["feasible"]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_segment_collision(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [((15.0, 15.0), (45.0, 45.0), False, "Diagonal through restricted zone"),
                    ((60.0, 30.0), (80.0, 50.0), True, "Diagonal in free space"),
                    ((10.0, 50.0), (70.0, 50.0), True, "Horizontal across free space"),
                    ((20.0, 30.0), (40.0, 30.0), False, "Segment through restricted zone"),
                    ((55.0, 55.0), (65.0, 65.0), False, "Segment near occupied point"),
                    ((70.0, 10.0), (70.0, 50.0), True, "Vertical in free space"),
                    ((25.0, 15.0), (25.0, 35.0), False, "Vertical through restricted zone"),]
        for start, end, expected, description in test_cases:
            x1, y1 = start
            x2, y2 = end
            result, _ = self.model.planner.check_segment_collision(x1, y1, x2, y2, geom=agent_geom)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_cost_computation(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [([(60.0, 40.0), (70.0, 50.0), (80.0, 60.0)], True, "Simple 3-point path in free space"),
                        ([(5.0, 5.0), (25.0, 25.0), (75.0, 75.0)], False, "Path crossing multiple obstacles"),
                        ([(50.0, 30.0), (60.0, 40.0)], True, "Short 2-point path in free space"),]
        for path, expected, description in test_cases:
            result = self.model.planner.compute_path_cost(path, geom=agent_geom)
            is_valid = (result != float("inf"))
            with self.subTest(msg=description):
                self.assertEqual(is_valid, expected, description)

    def test_path_validation(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [([(60.0, 40.0), (70.0, 50.0), (80.0, 60.0)], True, "Valid 3-point path in free space"),
                    ([(15.0, 55.0), (20.0, 60.0)], True, "Valid 2-point path"),
                    ([], False, "Empty path"),
                    ([(30.0, 30.0)], False, "Single point (path too short)"),
                    ([(20.0, 20.0), (30.0, 30.0)], False, "Path through restricted zone"),
                    ([(55.0, 55.0), (65.0, 65.0)], False, "Path near occupied point"),
                    ([(65.0, 20.0), (75.0, 30.0), (85.0, 40.0)], True, "Valid path in clear corner"),
                    ([(10.0, 10.0), (50.0, 50.0), (80.0, 80.0)], False, "Path crossing obstacles"),]
        for path, expected, description in test_cases:
            result = self.model.planner.validate_path(path, geom=agent_geom)[0]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)


class TestShapeAgentGeomEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_circle_geom_test_model()

    def test_collision(self):
        agent_geom = self.model.env.geoms['agent']
        test_cases = [((20.0, 20.0), True, "Free space - circle in clear area"),
                    ((15.0, 15.0), False, "Circle overlaps restricted zone"),
                    ((25.0, 25.0), True, "Circle interior restricted zone"),
                    ((63.0, 63.0), False, "Circle overlaps obstacle point"),
                    ((71.0, 20.0), False, "Circle barely overlaps obstacle point"),
                    ((70.0, 70.0), True, "Circle clear of obstacles"),
                    ((75.0, 75.0), True, "Circle in free space far from obstacles"),
                    ((15.0, 20.0), False, "Circle partially in restricted zone"),
                    ((80.0, 80.0), True, "Circle in far corner free space"),]
        for (x, y), expected, description in test_cases:
            result = self.model.planner.check_collision(x, y, geom=agent_geom)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_goal_feasibility(self):
        agent_geom = self.model.env.geoms['agent']
        test_cases = [((75.0, 75.0), True, "Goal in clear corner"),
                    ((50.0, 50.0), True, "Goal away from obstacles"),
                    ((13.0, 13.0), False, "Goal in restricted zone"),
                    ((20.0, 20.0), True, "Goal in center of restricted zone"),
                    ((16.0, 16.0), False, "Goal causes agent to overlap restricted zone"),
                    ((60.0, 60.0), False, "Goal at obstacle point"),
                    ((5.0, 5.0), True, "Goal at start area"),
                    ((71.0, 20.0), False, "Goal overlaps obstacle"),
                    ((40.0, 60.0), True, "Goal in free space"),]
        for goal, expected, description in test_cases:
            result = self.model.planner.check_goal_feasible(goal, geom=agent_geom)["feasible"]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_segment_collision(self):
        agent_geom = self.model.env.geoms['agent']
        test_cases = [((20.0, 20.0), (60.0, 60.0), False, "Diagonal through restricted zone"),
                    ((60.0, 60.0), (75.0, 75.0), False, "Diagonal through obstacle"),
                    ((44.0, 41.0), (75.0, 41.0), True, "Horizontal away from obstacles"),
                    ((20.0, 20.0), (30.0, 20.0), True, "Segment through restricted zone interior"),
                    ((50.0, 50.0), (60.0, 52.0), True, "Segment near obstacle"),
                    ((70.0, 20.0), (70.0, 60.0), False, "Vertical through obstacle region"),
                    ((41.0, 50.0), (60.0, 50.0), True, "Horizontal clear segment"),]
        for start, end, expected, description in test_cases:
            x1, y1 = start
            x2, y2 = end
            result, _ = self.model.planner.check_segment_collision(x1, y1, x2, y2, geom=agent_geom)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_cost_computation(self):
        agent_geom = self.model.env.geoms['agent']
        test_cases = [([(60.0, 90.0), (70.0, 80.0), (80.0, 70.0)], True, "Simple 3-point path in free space"),
                        ([(5.0, 5.0), (25.0, 25.0), (75.0, 75.0)], False, "Path crossing multiple obstacles"),
                        ([(50.0, 30.0), (60.0, 40.0)], True, "Short 2-point path in free space"),
                        ([(60.0, 10.0), (80.0, 30.0)], False, "Path through geom obstacle"),]
        for path, expected, description in test_cases:
            result = self.model.planner.compute_path_cost(path, geom=agent_geom)
            is_valid = (result != float("inf"))
            with self.subTest(msg=description):
                self.assertEqual(is_valid, expected, description)

    def test_path_validation(self):
        agent_geom = self.model.env.geoms['agent']
        test_cases = [([(60.0, 60.0), (70.0, 70.0), (75.0, 75.0)], False, "3-point path through obstacle"),
                    ([(50.0, 50.0), (55.0, 55.0)], True, "Valid 2-point path"),
                    ([], False, "Empty path"),
                    ([(20.0, 20.0)], False, "Single point (path too short)"),
                    ([(15.0, 15.0), (30.0, 30.0)], False, "Path through restricted zone"),
                    ([(60.0, 60.0), (65.0, 65.0)], False, "Path near obstacle"),
                    ([(60.0, 10.0), (71.0, 20.0)], False, "Path partially overlaps obstacle"),
                    ([(45.0, 50.0), (55.0, 65.0), (70.0, 80.0)], True, "Valid path away from obstacles"),
                    ([(10.0, 10.0), (50.0, 50.0)], False, "Path crossing obstacles"),]
        for path, expected, description in test_cases:
            result = self.model.planner.validate_path(path, geom=agent_geom)[0]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)


class TestShapeAgentHybridEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_circle_hybrid_test_model()

    def test_collision(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [((30.0, 50.0), True, "Circle in grid free space"),
                    ((25.0, 25.0), False, "Circle overlaps grid restricted zone"),
                    ((60.0, 60.0), False, "Circle overlaps geom obstacle"),
                    ((70.0, 50.0), True, "Circle clear of both grid and geom obstacles"),
                    ((16.0, 16.0), False, "Circle overlaps grid obstacle"),
                    ((74.0, 24.0), False, "Circle overlaps geom obstacle"),
                    ((50.0, 70.0), True, "Circle in hybrid free space"),
                    ((80.0, 80.0), False, "Circle overlaps grid occupied point"),]
        for (x, y), expected, description in test_cases:
            result = self.model.planner.check_collision(x, y, geom=agent_geom)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_goal_feasibility(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [((75.0, 60.0), True, "Goal in clear hybrid space"),
                    ((50.0, 70.0), True, "Goal away from both obstacles"),
                    ((25.0, 25.0), False, "Goal in grid obstacle"),
                    ((60.0, 60.0), False, "Goal overlaps geom obstacle"),
                    ((70.0, 50.0), True, "Goal in corner free space"),
                    ((30.0, 30.0), False, "Goal in grid zone obstacle"),
                    ((80.0, 35.0), True, "Goal far from obstacles"),]
        for goal, expected, description in test_cases:
            result = self.model.planner.check_goal_feasible(goal, geom=agent_geom)["feasible"]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_segment_collision(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [((30.0, 30.0), (60.0, 60.0), False, "Diagonal through grid obstacle"),
                    ((60.0, 40.0), (75.0, 50.0), True, "Diagonal in clear area"),
                    ((20.0, 50.0), (50.0, 50.0), True, "Horizontal in free space"),
                    ((25.0, 25.0), (35.0, 35.0), False, "Through grid restricted zone"),
                    ((55.0, 55.0), (70.0, 70.0), False, "Near geom obstacle"),
                    ((75.0, 40.0), (85.0, 50.0), True, "Clear segment in corner"),]
        for start, end, expected, description in test_cases:
            x1, y1 = start
            x2, y2 = end
            result, _ = self.model.planner.check_segment_collision(x1, y1, x2, y2, geom=agent_geom)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_cost_computation(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [([(60.0, 40.0), (70.0, 50.0), (80.0, 60.0)], True, "Simple 3-point path in free space"),
                        ([(5.0, 5.0), (25.0, 25.0), (75.0, 75.0)], False, "Path crossing multiple obstacles"),
                        ([(50.0, 30.0), (60.0, 40.0)], True, "Short 2-point path in free space"),
                        ([(60.0, 10.0), (80.0, 30.0)], False, "Path through geom obstacle"),]
        for path, expected, description in test_cases:
            result = self.model.planner.compute_path_cost(path, geom=agent_geom)
            is_valid = (result != float("inf"))
            with self.subTest(msg=description):
                self.assertEqual(is_valid, expected, description)

    def test_path_validation(self):
        agent_geom = self.model.geom_arch.geoms['agent']
        test_cases = [([(60.0, 40.0), (70.0, 50.0), (80.0, 60.0)], True, "Valid 3-point path in clear area"),
                    ([(50.0, 70.0), (65.0, 85.0)], True, "Valid 2-point path"),
                    ([], False, "Empty path"),
                    ([(30.0, 30.0)], False, "Single point (path too short)"),
                    ([(20.0, 20.0), (30.0, 30.0)], False, "Path through grid obstacle"),
                    ([(55.0, 55.0), (70.0, 70.0)], False, "Path near geom obstacle"),
                    ([(70.0, 40.0), (80.0, 50.0), (85.0, 60.0)], True, "Valid path in corner"),
                    ([(10.0, 10.0), (50.0, 50.0), (80.0, 80.0)], False, "Path through multiple obstacles"),]
        for path, expected, description in test_cases:
            result = self.model.planner.validate_path(path, geom=agent_geom)[0]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

if __name__ == '__main__':
    unittest.main()

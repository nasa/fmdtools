from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.define.object.geom import GeomParameter, GeomPoly, GeomPoint
from fmdtools.define.pathplan import PathPlannerState, PathPlannerParameter, PathPlannerBase


class BasePlannerParameter(PathPlannerParameter):
    start_point: tuple = (10.0, 10.0)
    end_point: tuple = (100.0, 100.0)

# --- Utility Function to Initialize Planner ---

def setup_planner(arch, environment):
    arch.add_fxn('planner', PathPlannerBase, p={'max_distance': arch.p.max_distance,
        'path_validation_enabled': arch.p.path_validation_enabled,
        'replanning_enabled': arch.p.replanning_enabled,
        'max_replan_attempts': arch.p.max_replan_attempts})
    arch.fxns['planner'].init_environment(environment)



#--- Grid Environment ---
class GridEnvironmentParam(CoordsParam):
    """Grid environment parameters."""
    blocksize: float = 10.0

class GridEnvironment(Coords):
    container_p = GridEnvironmentParam

    feature_traversable = (bool, True)
    feature_goal_allowed = (bool, True)
    feature_cost = (float, 1.0)

    def init_properties(self, **kwargs):
        self.set_range("traversable", False, xmin=20, xmax=40, ymin=20, ymax=40, inclusive=False)
        self.set_range("goal_allowed", False, xmin=20, xmax=40, ymin=20, ymax=40, inclusive=False)
        self.set_range("cost", 100.0, xmin=20, xmax=40, ymin=20, ymax=40, inclusive=False)
        
        occupied_cells = [(60, 60), (80, 80)]
        self.set_pts(occupied_cells, "traversable", False)
        self.set_pts(occupied_cells, "goal_allowed", False)
        self.set_pts(occupied_cells, "cost", float('inf'))

class GridArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    __slots__ = ('_grid',)
    
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.block_size) + 1
        y_size = int(self.p.end_point[1] / self.p.block_size) + 1
        self._grid = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.block_size))
        setup_planner(self, self._grid)
    @property
    def planner(self):
        """Get the planner function block."""
        return self.fxns['planner']

    @property
    def env(self):
        """Get the grid environment."""
        return self._grid



# --- Geom Environment ---
class ObstacleState(State):
    cost: float = 1.0
    traversable: bool = True
    goal_allowed: bool = False

class RestrictedZoneParam(GeomParameter):
    shell: tuple = ((10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0))
    holes: tuple = (((15.0, 15.0), (35.0, 15.0), (35.0, 35.0), (15.0, 35.0)),)

class RestrictedZoneGeom(GeomPoly):
    container_p = RestrictedZoneParam
    container_s = ObstacleState
    def get_shapely_args(self):     #override due to fmdtools bug
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

class ObstacleGeomArch(GeomArchitecture):
    container_p = BasePlannerParameter
    
    def init_architecture(self, **kwargs):
        self.add_geom("restricted_zone", RestrictedZoneGeom, s={'cost': 100.0, 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_60_60", OccupiedPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_70_20", SecondPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})

class GeomEnvironmentArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    default_sp = {'end_time': 15}
    __slots__ = ('_geom_arch',)

    def init_architecture(self, **kwargs):
        """Initialize architecture with flows and geometry obstacles."""
        self._geom_arch = ObstacleGeomArch()
        setup_planner(self, self._geom_arch)

    @property
    def geom_arch(self):
        """Get the geometry architecture containing all obstacles."""
        return self._geom_arch

    @property
    def geoms(self):
        """Get all obstacle geometries from the GeomArchitecture."""
        return self._geom_arch.geoms

    @property
    def planner(self):
        """Get the planner function block."""
        return self.fxns['planner']


# --- Hybrid Environment ---

class HybridEnvironment:
    def __init__(self, grid_env, geom_arch):
        self.grid = grid_env
        self.geom_arch = geom_arch
        self.geoms = geom_arch.geoms
    def to_index(self, x, y):
        return self.grid.to_index(x, y)
    @property
    def grid_property(self):
        return self.grid.grid

class HybridArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    default_sp = {'end_time': 15}  
    __slots__ = ('_grid', '_geom_arch', '_hybrid_env')
    
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.block_size) + 1
        y_size = int(self.p.end_point[1] / self.p.block_size) + 1
        self._grid = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.block_size))
        self._geom_arch = ObstacleGeomArch()
        self._hybrid_env = HybridEnvironment(self._grid, self._geom_arch)
        setup_planner(self, self._hybrid_env)
    @property
    def planner(self):
        """Get the planner function block."""
        return self.fxns['planner']

    @property
    def env(self):
        """Get the hybrid environment."""
        return self._hybrid_env
    
    @property
    def geoms(self):
        """Get all obstacle geometries."""
        return self.geom_arch.geoms


# --- Circular Agent ---
class CircleAgentParameter(PathPlannerParameter):
    start_point: tuple = (10.0, 10.0)
    end_point: tuple = (90.0, 90.0)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "block_size", 10.0)
        object.__setattr__(self, "agent_radius", 2.5)



class AgentState(State):
    cost: float = 0.0
    traversable: bool = True
    goal_allowed: bool = True

class AgentCircleParam(GeomParameter):
    coordinates: tuple = (0.0, 0.0) 
    buffer_around: float = 2.5

class AgentCircleGeom(GeomPoint):
    container_p = AgentCircleParam
    container_s = AgentState

class CircleAgentObstacleGeomArch(GeomArchitecture):
    container_p = BasePlannerParameter
    def init_architecture(self, **kwargs):
        self.add_geom("agent", AgentCircleGeom, p={'buffer_around': self.p.agent_radius}, s={'traversable': True, 'goal_allowed': True})
        self.add_geom("restricted_zone", RestrictedZoneGeom, s={'cost': 100.0, 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_60_60", OccupiedPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})
        self.add_geom("obstacle_70_20", SecondPointGeom, s={'cost': float('inf'), 'traversable': False, 'goal_allowed': False})

class CircleAgentGeomEnvironmentArchitecture(FunctionArchitecture):
    container_p = BasePlannerParameter
    default_sp = {'end_time': 15}
    __slots__ = ('_geom_arch',)
    def init_architecture(self, **kwargs):
        self._geom_arch = CircleAgentObstacleGeomArch()
        setup_planner(self, self._geom_arch)
    @property
    def geom_arch(self):
        """Get the geometry architecture containing all obstacles and agent."""
        return self._geom_arch
    
    @property
    def geoms(self):
        """Get all geometries (including agent) from the GeomArchitecture."""
        return self._geom_arch.geoms
    
    @property
    def planner(self):
        """Get the planner function block."""
        return self.fxns['planner']
    
    @property
    def agent_geom(self):
        """Get the circular agent geometry object."""
        return self._agent_geom
    
    @property
    def agent_shape(self):
        """Get the circular agent shape (shapely object)."""
        return self._agent_shape

from shapely import Point as ShapelyPoint
class CircleGridArchitecture(FunctionArchitecture):
    container_p = CircleAgentParameter
    __slots__ = ('_grid', '_agent_geom', '_agent_shape',)
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.block_size) + 1
        y_size = int(self.p.end_point[1] / self.p.block_size) + 1
        self._grid = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.block_size))
        setup_planner(self, self._grid)
        self.agent_shape = ShapelyPoint(0, 0).buffer(self.p.agent_radius)
    @property
    def planner(self):
        """Get the planner function block."""
        return self.fxns['planner']

    @property
    def env(self):
        """Get the grid environment."""
        return self._grid
    
    @property
    def agent_geom(self):
        """Get the circular agent geometry object."""
        return self._agent_geom
    
    @property
    def agent_shape(self):
        """Get the circular agent shape (shapely object)."""
        return self._agent_shape

class CircleGeomArchitecture(FunctionArchitecture):
    container_p = CircleAgentParameter
    def init_architecture(self, **kwargs):
        self._geom_arch = CircleAgentObstacleGeomArch()
        setup_planner(self, self._geom_arch)
        self.agent_shape = ShapelyPoint(0, 0).buffer(self.p.agent_radius)
    @property
    def geom_arch(self): return self._geom_arch
       
    @property
    def geoms(self):
        """Get all geometries (including agent) from the GeomArchitecture."""
        return self._geom_arch.geoms
    
    @property
    def planner(self):
        """Get the planner function block."""
        return self.fxns['planner']
    
    @property
    def agent_geom(self):
        """Get the circular agent geometry object."""
        return self._agent_geom
    
    @property
    def agent_shape(self):
        """Get the circular agent shape (shapely object)."""
        return self._agent_shape

class CircleHybridArchitecture(FunctionArchitecture):
    container_p = CircleAgentParameter
    def init_architecture(self, **kwargs):
        x_size = int(self.p.end_point[0] / self.p.block_size) + 1
        y_size = int(self.p.end_point[1] / self.p.block_size) + 1
        self._grid = GridEnvironment(p=GridEnvironmentParam(x_size=x_size, y_size=y_size, blocksize=self.p.block_size))
        self._geom_arch = CircleAgentObstacleGeomArch()
        self._hybrid_env = HybridEnvironment(self._grid, self._geom_arch)
        setup_planner(self, self._hybrid_env)
        self.agent_shape = ShapelyPoint(0, 0).buffer(self.p.agent_radius)
    @property
    def planner(self):
        """Get the planner function block."""
        return self.fxns['planner']

    @property
    def env(self):
        """Get the hybrid environment."""
        return self._hybrid_env
    
    @property
    def geom_arch(self):
        """Get the geometry architecture."""
        return self._geom_arch
    
    @property
    def geoms(self):
        """Get all geometries (including agent) from the architecture."""
        return self.geom_arch.geoms
    
    @property
    def agent_geom(self):
        """Get the circular agent geometry object."""
        return self._agent_geom
    
    @property
    def agent_shape(self):
        """Get the circular agent shape (shapely object)."""
        return self._agent_shape


def create_grid_test_model(start=(10.0, 10.0), end=(100.0, 100.0), block_size=10.0):
    """Create a standard grid-based test model."""
    return GridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'block_size': block_size,
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

def create_hybrid_test_model(start=(10.0, 10.0), end=(100.0, 100.0), block_size=10.0):
    """Create a hybrid (grid + geometry) test model."""
    return HybridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'block_size': block_size,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })

def create_circle_grid_test_model(start=(10.0, 10.0), end=(90.0, 90.0), agent_radius=2.5, block_size=10.0):
    """Create a circular agent grid-based test model."""
    return CircleGridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'block_size': block_size,
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

def create_circle_hybrid_test_model(start=(10.0, 10.0), end=(90.0, 90.0), agent_radius=2.5, block_size=10.0):
    """Create a circular agent hybrid (grid + geometry) test model."""
    return CircleHybridArchitecture(p={
        'start_point': start,
        'end_point': end,
        'block_size': block_size,
        'agent_radius': agent_radius,
        'path_validation_enabled': True,
        'replanning_enabled': True,
        'max_replan_attempts': 3,
        'max_distance': 5.0
    })


#####################################################################################################
import unittest

# --- Import your environment creation functions here ---
#from test_env import create_grid_test_model, create_geom_test_model, create_hybrid_test_model
#from integrate_env import create_grid_test_model, create_geom_test_model, create_hybrid_test_model

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
            result, _ = self.model.planner.check_segment_collision(*start, *end)
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
            result, _ = self.model.planner.check_segment_collision(*start, *end)
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
            result, _ = self.model.planner.check_segment_collision(*start, *end)
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

'''
from test_env_fine_grid import create_fine_grid_test_model

class TestFineGridEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_fine_grid_test_model()

    def test_segment_collision(self):
        test_cases = [((1.0, 1.0), (15.0, 15.0), False, "Long diagonal through obstacles"),
                    ((1.0, 1.0), (5.0, 1.0), True, "Horizontal clear segment"),
                    ((11.0, 1.0), (15.0, 1.0), True, "Horizontal in clear area"),
                    ((6.0, 7.5), (6.0, 9.5), False, "Vertical through narrow corridor"),
                    ((7.0, 7.5), (7.0, 9.5), False, "Vertical intersecting corridor"),
                    ((10.5, 7.5), (10.5, 9.5), True, "Vertical outside corridor"),
                    ((2.0, 2.0), (3.5, 3.5), False, "Diagonal through small obstacle"),
                    ((4.0, 4.0), (6.0, 6.0), True, "Diagonal avoiding obstacles"),
                    ((2.0, 12.0), (20.0, 2.0), False, "Long diagonal crossing obstacles"),
                    ((10.0, 16.0), (18.0, 16.0), True, "Clear horizontal at distance"),
                    ((0.5, 0.5), (2.0, 2.0), False, "Short segment near start through obstacle"),
                    ((5.0, 15.0), (15.0, 19.5), True, "Short segment far corner clear"),]
        for start, end, expected, description in test_cases:
            result, _ = self.model.planner.check_segment_collision(*start, *end)
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_path_validation(self):
        test_cases = [([(1.0, 1.0), (5.0, 1.0), (10.0, 10.0)], False, "Path crossing obstacles"),
                    ([(1.0, 1.0), (1.0, 4.0), (4.0, 4.0)], True, "Short valid path in start area"),
                    ([(15.0, 10.0), (18.0, 15.0), (19.0, 16.0)], True, "Valid path in far corner"),
                    ([(10.5, 7.5), (10.5, 9.5)], True, "Valid 2-point path outside corridor"),
                    ([(6.0, 7.5), (6.0, 9.5)], False, "Path through narrow corridor"),
                    ([(1.0, 15.0), (15.0, 1.0)], False, "Long diagonal crossing obstacles"),
                    ([(2.0, 2.0), (3.5, 3.5), (5.0, 5.0)], False, "Path through obstacle region"),
                    ([], False, "Empty path"),
                    ([(15.0, 15.0)], False, "Single point (too short)"),]
        for path, expected, description in test_cases:
            result = self.model.planner.validate_path(path)[0]
            with self.subTest(msg=description):
                self.assertEqual(result, expected, description)

    def test_dda_use(self):
        pass
'''

''' NEED TO ADD IN THE SHAPE PART
from integrate_env import create_circle_grid_test_model, create_circle_geom_test_model, create_circle_hybrid_test_model

class TestShapeAgentGridEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = create_circle_grid_test_model()

    def test_collision(self):
        test_cases = [((70.0, 10.0), True, "Free space - circle center safely in free space"),
                    ((25.0, 25.0), False, "Circle overlaps with restricted zone boundary"),
                    ((30.0, 30.0), False, "Circle center in restricted zone"),
                    ((50.0, 50.0), True, "Circle center away from all obstacles"),
                    ((60.0, 60.0), False, "Circle overlaps with occupied point"),
                    ((70.0, 70.0), True, "Circle clear of obstacles"),
                    ((80.0, 80.0), False, "Circle at occupied point"),
                    ((10.0, 10.0), True, "Circle at start area"),]
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
        test_cases = [(15.0, 15.0), (45.0, 45.0), False, "Diagonal through restricted zone"),
                        (60.0, 30.0), (80.0, 50.0), True, "Diagonal in free space"),
                        ((10.0, 50.0, 70.0, 50.0), True, "Horizontal across free space"),
                        ((20.0, 30.0, 40.0, 30.0), False, "Segment through restricted zone"),
                        ((55.0, 55.0, 65.0, 65.0), False, "Segment near occupied point"),
                        ((70.0, 10.0, 70.0, 50.0), True, "Vertical in free space"),
                        ((25.0, 15.0, 25.0, 35.0), False, "Vertical through restricted zone"),]
        for start, end, expected, description in test_cases:
            result, _ = self.model.planner.check_segment_collision(*start, *end)
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
'''

if __name__ == '__main__':
    unittest.main()

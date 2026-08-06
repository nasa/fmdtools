from fmdtools.define.container.state import State
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.object.geom import GeomPoint, GeomParameter, GeomLine, GeomPoly
from fmdtools.define.architecture.geom import GeomArchitecture
from fmdtools.analyze.common import calc_metric
from fmdtools.define.object.base import BaseObject

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import traceback
import heapq
from shapely.geometry import Point as ShapelyPoint  ##NOTE: maybe use GeomPoint instead
from shapely.geometry import LineString as ShapelyLineString    ##NOTE: maybe use GeomLine instead
from shapely.geometry import box as shapely_box
from shapely.affinity import translate

class PathPlannerState(State):
    """
    Generic planner state:
    - flightplan: tuple of (x,y) points
    - planned: whether the plan is valid
    - pt: current index within plan
    - last_valid_path: stores last validated path for recovery
    - replanning_triggered: flag indicating if replanning was needed
    """
    flightplan: tuple = ()
    planned: bool = False
    pt: int = 0
    last_valid_path: tuple = ()
    replanning_triggered: bool = False


class PathPlannerParameter(Parameter):
    """
    Static planner parameters (set at initialization).
    """
    max_distance: float = 5.0
    blocksize: float = 2.5
    agent_radius: float = 0.0 # Optional - agent's circular radius
    planner = None
    
    # Validation parameters
    path_validation_enabled: bool = True
    replanning_enabled: bool = True
    max_replan_attempts: int = 3
    collision_check_resolution: float = 0.5



class PathPlannerBase(BaseObject):
    """
    Universal base path planner for both Coords (grid) and Geom/GeomArchitecture (continuous)
    environments.

    Users can supply a custom cost function via:
        planner.cost_function = fn

    where fn is one of:
        fn(path, planner)  -> float
        fn(path)           -> float
        fn(segment_costs)  -> float
    """

    __slots__ = ('env', 'cost_function',
                 'computation_history', 'rejected_paths', 'replan_reasons',)

    # Container references
    container_s = PathPlannerState   # PathPlannerState
    container_p = PathPlannerParameter   # PathPlannerParameter
    #container_m = None   # Mode

    def __init__(self, cost_function=None, **kwargs):
        super().__init__(**kwargs)
        self.env = None
        self.cost_function = None #have the user set externally

        self.computation_history = []
        self.rejected_paths = []
        self.replan_reasons = []


# --- Environment initialization ---    
    def init_environment(self, env):
        """
        Initialize with either Coords, Geom, or GeomArchitecture.
        """
        if env is None:
            raise ValueError("Environment cannot be None.")

        self.env = env

    @property
    def get_env(self):
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call init_environment() first.")
        return self.env


# --- Environment type detection ---
    def is_coords(self):
        """
        Returns True if current environment is Coords.

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGrid())
        >>> pp.is_coords()
        True
        >>> pp.init_environment(ExampleGeomArch())
        >>> pp.is_coords()
        False
        >>> pp.init_environment(ExampleHybrid())
        >>> pp.is_coords()
        False
        """
        return isinstance(self.env, Coords)

    def is_geom_arch(self):
        """
        Returns True if current environment is GeomArchitecture.

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGeomArch())
        >>> pp.is_geom_arch()
        True
        >>> pp.init_environment(ExampleGrid())
        >>> pp.is_geom_arch()
        False
        >>> pp.init_environment(ExampleHybrid())
        >>> pp.is_geom_arch()
        False
        """
        return isinstance(self.env, GeomArchitecture)
    
    def is_hybrid(self):
        """
        Returns True if environment is hybrid (has both grid and geoms).

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleHybrid())
        >>> pp.is_hybrid()
        True
        >>> pp.init_environment(ExampleGrid())
        >>> pp.is_hybrid()
        False
        """
        return (hasattr(self.env, 'grid') and 
            hasattr(self.env, 'geom_arch') and 
            isinstance(self.env.grid, Coords) and 
            isinstance(self.env.geom_arch, GeomArchitecture))

    def _get_geom_arch(self):
        """
        Get the GeomArchitecture from the environment.
        
        Returns
        -------
        GeomArchitecture or None
            The geometry architecture, or None if not available
        
        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGeomArch())
        >>> pp._get_geom_arch() is pp.env
        True
        >>> pp.init_environment(ExampleHybrid())
        >>> isinstance(pp._get_geom_arch(), ExampleGeomArch)
        True
        >>> pp.init_environment(ExampleGrid())
        >>> pp._get_geom_arch() is None
        True
        """
        if self.is_geom_arch():
            return self.env
        elif self.is_hybrid():
            return self.env.geom_arch
        return None


    def get_buffered_shape(self, geom_obj, buffer_name=None):
        """
        Returns a buffered (footprint) shape from a Geom object for collision checking.
        Uses specified buffer_name, else first available buffer, else base shape.
        
        Parameters
        ----------
        geom_obj : Geom
            The geometry object (agent, obstacle, etc.)
        buffer_name : str, optional
            The specific buffer name to use (e.g., "on", "around", etc).
            If None, uses the first available buffer.
        Returns
        -------
        shape : shapely.geometry
            The shapely shape to use for collision checks.
        """
        try:
            if hasattr(geom_obj, 'create_shape'):
                buffer_attrs = getattr(geom_obj.p, 'get_pref_attrs', lambda x: {} )('buffer')
                if buffer_name is not None:
                    # Use specified buffer if present
                    if buffer_name in buffer_attrs:
                        return geom_obj.create_shape(buffer_name)
                if buffer_attrs:
                    # Use the first available buffer
                    first_buffer = list(buffer_attrs.keys())[0]
                    return geom_obj.create_shape(first_buffer)
                else:
                    # Fall back to base shape if no buffers defined
                    return geom_obj.create_shape()
        except Exception as e:
            print(f"Warning: Couldn't create buffered shape: {e}")
        return None


# --- Obstacle query for Coords ---
    ## TODO: I want to generalize the feature dictionary so the user can put in names, colors, etc.
    def _query_coords_cell(self, x, y):
        """
        Aggregate obstacle behavior from CoordsParam attributes.
        Each feature in coords.p.<name> must be a dict:
            {cost, traversable, goal_allowed}

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGrid())
        >>> pp._query_coords_cell(2,2)
        {'cost': inf, 'traversable': False, 'goal_allowed': False}
        >>> pp._query_coords_cell(9,9)
        {'cost': 1.0, 'traversable': True, 'goal_allowed': True}
        """
        grid_env = self.env.grid if self.is_hybrid() else self.env
        ## TODO: what if they didn't pass in the information
        traversable = grid_env.get(x, y, "traversable", outside=True)
        cost = grid_env.get(x, y, "cost", outside=1.0)
        goal_allowed = grid_env.get(x, y, "goal_allowed", outside=True)
        return {"cost": float(cost if traversable else float('inf')), 
                "traversable": bool(traversable), 
                "goal_allowed": bool(goal_allowed),}


# --- Obstacle query for Geom / GeomArchitecture ---
    def _query_geom_point(self, x, y):
        """
        Query a point against all geometries using GeomArchitecture.all_at().

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGeomArch())
        >>> # Inside the small circle at (8,8)
        >>> pp._query_geom_point(8,8)
        {'cost': 0.0, 'traversable': False, 'goal_allowed': False}
        >>> # Outside the circle
        >>> pp._query_geom_point(1,1)
        {'cost': 0.0, 'traversable': True, 'goal_allowed': True}
        """
        total_cost = 0.0
        traversable = True
        goal_allowed = True

        # Get the GeomArchitecture
        geom_arch = self._get_geom_arch()
        if geom_arch is None:
            return {"cost": 0.0, "traversable": True, "goal_allowed": True}

        geoms_at_point = geom_arch.all_at(x, y)
        
        # Process all geoms that contain this point
        for geom_name in geoms_at_point:
            geom_obj = geom_arch.geoms.get(geom_name)
            
            if geom_obj is None or not hasattr(geom_obj, 's'):
                continue
            
            # Extract properties from state
            cost = getattr(geom_obj.s, "cost", 0.0)
            trav = getattr(geom_obj.s, "traversable", True)
            goal = getattr(geom_obj.s, "goal_allowed", True)

            # Update aggregate values
            if not trav:
                traversable = False
            if not goal:
                goal_allowed = False
            if trav:
                total_cost += cost

        return {"cost": total_cost, "traversable": traversable, "goal_allowed": goal_allowed}
    

# --- Unified point query ---
    def query_point(self, x, y):
        """
        Unified hybrid environment point query.
        Supports: Coords only, Geom only, or BOTH simultaneously

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGrid())
        >>> # Not traversable, not goal_allowed
        >>> pp.query_point(2,2)
        {'cost': inf, 'traversable': False, 'goal_allowed': False}
        >>> # Traversable region
        >>> pp.query_point(9,9)
        {'cost': 1.0, 'traversable': True, 'goal_allowed': True}
        >>> pp.init_environment(ExampleGeomArch())
        >>> # Inside the small circle at (8,8)
        >>> pp.query_point(8,8)
        {'cost': inf, 'traversable': False, 'goal_allowed': False}
        >>> # Outside the circle
        >>> pp.query_point(1,1)
        {'cost': 0.0, 'traversable': True, 'goal_allowed': True}
        >>> pp.init_environment(ExampleHybrid())
        >>> # Hybrid at (8,8): both grid and geom allow
        >>> pp.query_point(8,8)
        {'cost': inf, 'traversable': False, 'goal_allowed': False}
        >>> # Only grid (traversable/goal_allowed for grid)
        >>> pp.query_point(9,9)
        {'cost': 1.0, 'traversable': True, 'goal_allowed': True}
        >>> # Only grid at a restricted region
        >>> pp.query_point(2,2)
        {'cost': inf, 'traversable': False, 'goal_allowed': False}
        """
        results = []

        # Check grid
        if self.is_coords() or self.is_hybrid():
            results.append(self._query_coords_cell(x, y))

        # Check geom obstacles (GeomArchitecture or Hybrid)
        if self.is_geom_arch() or self.is_hybrid():
            results.append(self._query_geom_point(x, y))

        # Merge results
        if not results:
            return {"cost": 0.0, "traversable": True, "goal_allowed": True}

        total_cost = sum(r["cost"] for r in results if r["traversable"])
        traversable = all(r["traversable"] for r in results)
        goal_allowed = all(r["goal_allowed"] for r in results)

        if not traversable:
            total_cost = float('inf')

        return {"cost": float(total_cost), 
                "traversable": bool(traversable), 
                "goal_allowed": bool(goal_allowed)}
    

    def check_collision(self, x, y, geom=None):
        """
        Returns True if traversable.
        - If shape is provided: acts as check_shape_collision.

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGrid())
        >>> pp.check_collision(2,2)
        False
        >>> pp.check_collision(9,9)
        True
        >>> pp.init_environment(ExampleGeomArch())
        >>> pp.check_collision(8,8)
        False
        >>> pp.check_collision(1,1)
        True
        """
        if geom is None:
            return self.query_point(x, y)["traversable"]
        return self.check_shape_collision(x, y, geom)[0]


    def check_goal_feasible(self, goal, geom=None):
        """
        Check whether goal is feasible for a shaped agent.
        
        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.init_environment(ExampleGrid())
        >>> # Not traversable (2,2) and not goal_allowed
        >>> pp.check_goal_feasible((2,2))['feasible']
        False
        >>> # Traversable and goal_allowed region!
        >>> pp.check_goal_feasible((9,9))['feasible']
        True
        >>> pp.init_environment(ExampleGeomArch())
        >>> # Inside the small circle at (8,8): not traversable and not goal_allowed
        >>> pp.check_goal_feasible((8,8))['feasible']
        False
        >>> # Outside circle: traversable, goal_allowed
        >>> pp.check_goal_feasible((1,1))['feasible']
        True
        """
        x, y = goal
        
        if geom is not None:
            # Use shape-aware collision
            is_free, collision_info = self.check_shape_collision(x, y, geom)
            if not is_free:
                return {"feasible": False,
                        "reason": "Goal position causes agent to collide with obstacles.",
                        "collision_info": collision_info}
        else:
            # Fallback to point query
            info = self.query_point(x, y)
            if not info["traversable"] or not info["goal_allowed"]:
                return {"feasible": False,
                        "reason": "Goal point is not in traversable/goal-allowed region.",
                        "point_info": info}
        
        return {"feasible": True, "reason": "Goal point is valid."}

    
# --- Segment collision checking ---
    def check_segment_collision(self, x1, y1, x2, y2, geom=None, shape_name="shape", resolution=None):
        """
        Check if a line segment from (x1, y1) to (x2, y2) is collision-free.
        
        If geom is provided, performs a swept-volume check (shape along segment).
        If geom is None, performs a point-based check.
        
        Parameters
        ----------
        x1, y1, x2, y2 : float
            Segment endpoints
        geom : GeomPoint/GeomPoly/GeomLine, optional
            The agent shape. If None, treats agent as a point.
        shape_name : str
            Name for get_buffered_shape lookup (only used when geom is provided)
        resolution : float, optional
            Distance between sample points for shape checks.
            If None, uses collision_check_resolution.
        
        Returns
        -------
        tuple (is_free, collision_info)
            is_free : bool
            collision_info : list of collision points (point mode) or dict (shape mode)
        """
        if geom is not None:
            result = self._query_segment_shape(x1, y1, x2, y2, geom,
                                            shape_name=shape_name,
                                            resolution=resolution)
            return result["traversable"], result["collision_info"]

        # --- Point-based collision check ---
        collision_points = []

        # For Coords (grid) environments
        if self.is_coords() or self.is_hybrid():
            grid_env = self.env.grid if self.is_hybrid() else self.env
            try:
                grid_x1, grid_y1 = grid_env.to_index(x1, y1)
                grid_x2, grid_y2 = grid_env.to_index(x2, y2)
            except:
                collision_points.append((x1, y1))
                collision_points.append((x2, y2))
                return False, collision_points

            
            cell_size = getattr(grid_env.p, 'blocksize', 1.0)
            if cell_size < 1.0:
                cells = self._dda_line(grid_x1, grid_y1, grid_x2, grid_y2)
            else:
                cells = self._bresenham_line(grid_x1, grid_y1, grid_x2, grid_y2)

            for gx, gy in cells:
                # Convert cell index back to a world coordinate (cell center).
                wx = gx * cell_size
                wy = gy * cell_size

                pt_info = self.query_point(wx, wy)

                if not pt_info["traversable"]:
                    collision_points.append((wx, wy))
                    return False, collision_points

        # For Geom / GeomArchitecture / Hybrid environments
        if self.is_geom_arch() or self.is_hybrid():
            segment = ShapelyLineString([(x1, y1), (x2, y2)])
            geom_arch = self._get_geom_arch()

            if geom_arch is not None:
                for geom_name, geom_obj in geom_arch.geoms.items():
                    if not hasattr(geom_obj, 's'):
                        continue

                    traversable = getattr(geom_obj.s, "traversable", True)
                    if traversable:
                        continue

                    shape = self.get_buffered_shape(geom_obj=geom_obj)
                    if shape is None:
                        continue

                    if segment.intersects(shape):
                        intersection = segment.intersection(shape)
                        collision_pt = self._extract_intersection_point(intersection)
                        if collision_pt:
                            collision_points.append(collision_pt)
                        return False, collision_points

        return True, collision_points

    def _extract_intersection_point(self, intersection):
        """
        Extract a representative point from a shapely geometry intersection.
        Handles Point, LineString, Polygon, and MultiPart geometries.
        """
        # Try coords attribute (Point or LineString)
        try:
            if hasattr(intersection, 'coords'):
                coords = list(intersection.coords)
                if coords:
                    return coords[0]
        except (NotImplementedError, AttributeError):
            pass
        
        # Try exterior (Polygon)
        try:
            if hasattr(intersection, 'exterior'):
                coords = list(intersection.exterior.coords)
                if coords:
                    return coords[0]
        except (NotImplementedError, AttributeError):
            pass
        
        # Try geoms (MultiPoint, MultiLineString, MultiPolygon)
        try:
            if hasattr(intersection, 'geoms'):
                geoms = list(intersection.geoms)
                if geoms:
                    # Recursively extract from first sub-geometry
                    return self._extract_intersection_point(geoms[0])
        except (NotImplementedError, AttributeError):
            pass
        
        # Fallback: use centroid
        try:
            if hasattr(intersection, 'centroid'):
                centroid = intersection.centroid
                return (centroid.x, centroid.y)
        except Exception:
            pass
        
        return (0.0, 0.0)


    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Generate all grid cells that a line segment passes through using Bresenham's algorithm.
        Works for integer grid coordinates only (faster).
        
        Returns
        -------
        list of (x, y) grid cell coordinates
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells


    def _dda_line(self, x0, y0, x1, y1):
        """
        Generate all grid cells that a line segment passes through using DDA.
        Works for floating-point grid coordinates.
        
        Parameters
        ----------
        x0, y0, x1, y1 : float
            Line endpoints as integer grid cell indices
        
        Returns
        -------
        list of unique (grid_x, grid_y) cell coordinates
        """
        cells = set()
        
        dx = x1 - x0
        dy = y1 - y0
        
        grid_x = int(x0)
        grid_y = int(y0)
        cells.add((grid_x, grid_y))
        
        if dx == 0 and dy == 0:
            return list(cells)
        
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        
        # Distance to next grid line (in parameter t space, 0 to 1)
        if dx != 0:
            t_max_x = (grid_x + (1 if dx > 0 else 0) - x0) / dx
            t_delta_x = 1.0 / abs(dx)
        else:
            t_max_x = float('inf')
            t_delta_x = float('inf')
        
        if dy != 0:
            t_max_y = (grid_y + (1 if dy > 0 else 0) - y0) / dy
            t_delta_y = 1.0 / abs(dy)
        else:
            t_max_y = float('inf')
            t_delta_y = float('inf')
        
        # Get the target cell (floor of endpoint)
        target_x = int(x1)
        target_y = int(y1)
        
        # Traverse grid cells until we reach the target cell
        while grid_x != target_x or grid_y != target_y:
            if t_max_x < t_max_y:
                t_max_x += t_delta_x
                grid_x += step_x
            else:
                t_max_y += t_delta_y
                grid_y += step_y
            
            cells.add((grid_x, grid_y))
        
        return list(cells)


# --- Collision checking when agent is a shape ---
    def _get_grid_cells_in_bounds(self, shape_bounds, grid_env):
        """
        Get all grid cells that overlap with a shape's bounding box.
        This is efficient for determining which cells need to be checked.
        
        Parameters
        ----------
        shape_bounds : tuple
            (minx, miny, maxx, maxy) from shapely geometry.bounds
        grid_env : Coords
            Grid environment
        
        Returns
        -------
        set of (grid_x, grid_y) tuples
            All grid cells overlapping the bounding box
        """
        minx, miny, maxx, maxy = shape_bounds
        
        # Convert to grid indices
        grid_x_min, grid_y_min = grid_env.to_index(minx, miny)
        grid_x_max, grid_y_max = grid_env.to_index(maxx, maxy)
        
        # Ensure proper ordering
        grid_x_min, grid_x_max = min(grid_x_min, grid_x_max), max(grid_x_min, grid_x_max)
        grid_y_min, grid_y_max = min(grid_y_min, grid_y_max), max(grid_y_min, grid_y_max)
        
        cells = set()
        for gx in range(grid_x_min, grid_x_max + 1):
            for gy in range(grid_y_min, grid_y_max + 1):
                cells.add((gx, gy))
        
        return cells

    def query_shape(self, shape):
        """
        Query properties of a region defined by a shapely geometry.
        Useful for checking if a shape-based agent can occupy a position.
        
        Parameters
        ----------
        shape : shapely.geometry.base.BaseGeometry
            Shape to query (e.g., Point, Polygon, Circle)
        
        Returns
        -------
        dict
            {"traversable": bool, "cost": float, "goal_allowed": bool, "blocked_cells": list}
        """
        traversable = True
        total_cost = 0.0
        goal_allowed = True
        blocked_cells = []

        # For Coords (grid) environments
        if self.is_coords() or self.is_hybrid():
            grid_env = self.env.grid if self.is_hybrid() else self.env
            
            # Get cells overlapping the shape's bounding box
            cells = self._get_grid_cells_in_bounds(shape.bounds, grid_env)
            
            for grid_x, grid_y in cells:
                cell_center = grid_env.grid[grid_x, grid_y]
                
                # Check if the shape overlaps with this cell
                # Use a small buffer around the center point to approximate cell coverage
                cell_size = getattr(grid_env.p, 'cell_size', 1.0)
                cell_buffer = ShapelyPoint(cell_center).buffer(cell_size * 0.5)     
                
                if shape.intersects(cell_buffer):
                    cell_info = self.query_point(*cell_center)
                    
                    if not cell_info["traversable"]:
                        traversable = False
                        blocked_cells.append((grid_x, grid_y))
                    
                    if cell_info["cost"] != 0.0:
                        total_cost += cell_info["cost"]
                    
                    if not cell_info["goal_allowed"]:
                        goal_allowed = False

        # For Geom / GeomArchitecture
        if self.is_geom_arch() or self.is_hybrid():
            geom_arch = self._get_geom_arch()
            if geom_arch is not None:
                for geom_name in geom_arch.geoms:
                    geom_obj = geom_arch.geoms[geom_name]
                    if not hasattr(geom_obj, 's'):
                        continue
                    
                    trav = getattr(geom_obj.s, "traversable", True)
                    cost = getattr(geom_obj.s, "cost", 0.0)
                    goal = getattr(geom_obj.s, "goal_allowed", True)
                    
                    # Check intersection using shapely
                    obstacle_shape = self.get_buffered_shape(geom_obj)
                    if obstacle_shape is not None:
                        if shape.intersects(obstacle_shape):
                            if not trav:
                                traversable = False
                            total_cost += cost
                            if not goal:
                                goal_allowed = False
                        

        return {"traversable": traversable,
                "cost": total_cost if traversable else float('inf'),
                "goal_allowed": goal_allowed,
                "blocked_cells": blocked_cells}


    def check_shape_collision(self, x, y, geom, shape_name="shape"):
        """
        Check if shape at (x,y) collides with obstacles.
        
        Parameters
        ----------
        x, y : float
            Position to check
        geom : GeomPoint/GeomPoly/GeomLine
            The agent shape (Geom object)
        
        Returns
        -------
        tuple (is_traversable, collision_info)
        """
        #print("check_shape_collision called")  # Already printing, so this works.
        if geom is None: # or self.p.agent_radius == 0.0:
            #print("Early return: no geom or zero radius")
            return self.query_point(x, y)["traversable"], {}

        shape = self.get_buffered_shape(geom, shape_name)
        if shape is None: # or self.p.agent_radius == 0.0:
            #print("Early return: failed to get buffered shape or zero radius")
            return self.query_point(x, y)["traversable"], {}

        collision_info = {"blocked_cells": [], "blocked_geoms": []}

        try:
            query_shape = translate(shape, xoff=x, yoff=y)
            #print("Translated shape:", query_shape, query_shape.bounds)
        except Exception as e:
            #print(f"Warning: Could not translate shape: {e}")
            return self.query_point(x, y)["traversable"], {}

        if self.is_geom_arch() or self.is_hybrid():
            #print("Entering geom collision check...")
            try:
                geom_arch = self._get_geom_arch()
                if geom_arch is not None:                    
                    for geom_name in geom_arch.geoms:
                        geom_obj = geom_arch.geoms[geom_name]
                        
                        # Skip if not a proper geometry object
                        if not hasattr(geom_obj, 's'):
                            continue
                        
                        # Get traversability from state
                        traversable = getattr(geom_obj.s, 'traversable', True)
                        if traversable:
                            continue
                        
                        obstacle_shape = self.get_buffered_shape(geom_obj=geom_obj)
                        if obstacle_shape is None:
                            #continue
                            print(f"Warning: could not build shape for obstacle '{geom_name}'; "
                              f"treating as collision (fail-closed).")
                            collision_info["blocked_geoms"].append(geom_name)
                            return False, collision_info
                        # Check intersection
                        #print("Agent shape:", query_shape, query_shape.bounds)
                        #print("Obstacle shape:", obstacle_shape, obstacle_shape.bounds)
                        if query_shape.intersects(obstacle_shape):
                            collision_info["blocked_geoms"].append(geom_name)
                            return False, collision_info
            except Exception as e:
                print(f"Warning: Geometry collision check failed: {e}")
                traceback.print_exc()

        # ===== GRID COLLISION CHECK =====
        if self.is_coords() or self.is_hybrid():
            grid_env = self.env.grid if self.is_hybrid() else self.env
            blocksize = getattr(grid_env.p, 'blocksize', 10.0)

            minx, miny, maxx, maxy = query_shape.bounds

            try:
                min_gx, min_gy = grid_env.to_index(minx, miny)
                max_gx, max_gy = grid_env.to_index(maxx, maxy)
            except Exception as e:
                print(f"Warning: Grid collision check failed (shape out of bounds): {e}")
                # Fail closed: if we can't verify the region (e.g., outside the grid),
                # treat it as blocked rather than silently assuming it's free.
                collision_info["blocked_cells"].append("out_of_bounds")
                return False, collision_info

            # Ensure proper ordering (bounds min/max don't guarantee index min/max)
            min_gx, max_gx = min(min_gx, max_gx), max(min_gx, max_gx)
            min_gy, max_gy = min(min_gy, max_gy), max(min_gy, max_gy)

            half = blocksize / 2

            for gx in range(min_gx, max_gx + 1):
                for gy in range(min_gy, max_gy + 1):
                    grid_x = gx * blocksize
                    grid_y = gy * blocksize

                    cell_square = shapely_box(grid_x - half, grid_y - half,
                                      grid_x + half, grid_y + half)

                    if not query_shape.intersects(cell_square):
                        continue

                    # This cell overlaps the agent -> check its properties.
                    info = self._query_coords_cell(grid_x, grid_y)

                    if not info["traversable"]:
                        collision_info["blocked_cells"].append((grid_x, grid_y))
                        return False, collision_info

        return True, collision_info


    def _query_segment_shape(self, x1, y1, x2, y2, geom, shape_name="shape", resolution=None):
        """
        Sweep a shape along a segment, sampling at regular intervals.
        Returns traversability, total accumulated cost, and collision info.

        This is the shared logic used by both:
        - check_segment_collision (collision check only)
        - compute_path_cost (cost accumulation for shape agents)

        Parameters
        ----------
        x1, y1, x2, y2 : float
            Segment endpoints (center of the agent shape)
        geom : GeomPoint/GeomPoly/GeomLine
            The agent shape (Geom object)
        shape_name : str
            Name for get_buffered_shape lookup
        resolution : float, optional
            Distance between sample points. If None, uses collision_check_resolution.

        Returns
        -------
        dict
            {
                "traversable": bool,
                "cost": float,          # total weighted cost along the segment
                "collision_info": dict,  # details if collision found
                "cells_visited": set,   # all (grid_x, grid_y) cells the shape overlapped
            }
        """
        if resolution is None:
            resolution = getattr(self.p, 'collision_check_resolution', 0.5)

        shape = self.get_buffered_shape(geom, shape_name)
        if shape is None:
            # Fallback to point-based check
            is_free, collision_pts = self.check_segment_collision(x1, y1, x2, y2)
            if not is_free:
                return {"traversable": False, "cost": float('inf'),
                        "collision_info": {"blocked_cells": [], "blocked_geoms": []},
                        "cells_visited": set()}
            # Point fallback cost
            pt1 = self.query_point(x1, y1)
            pt2 = self.query_point(x2, y2)
            c1 = pt1["cost"] if pt1["cost"] > 0 else 1.0
            c2 = pt2["cost"] if pt2["cost"] > 0 else 1.0
            avg = 0.5 * (c1 + c2)
            segdist = math.hypot(x2 - x1, y2 - y1)
            return {"traversable": True, "cost": avg * segdist,
                    "collision_info": {}, "cells_visited": set()}

        distance = math.hypot(x2 - x1, y2 - y1)
        num_samples = max(2, int(distance / resolution) + 1)
        step_dist = distance / (num_samples - 1) if num_samples > 1 else 0.0

        total_cost = 0.0
        cells_visited = set()
        collision_info = {"blocked_cells": [], "blocked_geoms": []}

        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.0
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            # Translate the agent shape to the sample position
            try:
                query_shape = translate(shape, xoff=x, yoff=y)
            except Exception:
                return {"traversable": False, "cost": float('inf'),
                        "collision_info": collision_info, "cells_visited": cells_visited}

            # --- Check grid cells (Coords / Hybrid) ---
            sample_cost = 0.0
            if self.is_coords() or self.is_hybrid():
                grid_env = self.env.grid if self.is_hybrid() else self.env
                cells = self._get_grid_cells_in_bounds(query_shape.bounds, grid_env)
                cell_size = getattr(grid_env.p, 'blocksize', 10.0)
                half = cell_size / 2.0

                for grid_x, grid_y in cells:
                    cell_center = grid_env.grid[grid_x, grid_y]
                    cx, cy = cell_center

                    cell_square = shapely_box(cx - half, cy - half,
                                            cx + half, cy + half)

                    if not query_shape.intersects(cell_square):
                        continue

                    info = self._query_coords_cell(cx, cy)

                    if not info["traversable"]:
                        collision_info["blocked_cells"].append((grid_x, grid_y))
                        return {"traversable": False, "cost": float('inf'),
                                "collision_info": collision_info,
                                "cells_visited": cells_visited}

                    cells_visited.add((grid_x, grid_y))
                    cell_cost = info.get("cost", 0.0)
                    if cell_cost > 0:
                        sample_cost += cell_cost

            # --- Check geom obstacles (GeomArchitecture / Hybrid) ---
            if self.is_geom_arch() or self.is_hybrid():
                geom_arch = self._get_geom_arch()
                if geom_arch is not None:
                    for geom_name, geom_obj in geom_arch.geoms.items():
                        if not hasattr(geom_obj, 's'):
                            continue
                        traversable = getattr(geom_obj.s, 'traversable', True)
                        if traversable:
                            # Could still have a cost associated
                            obs_shape = self.get_buffered_shape(geom_obj=geom_obj)
                            if obs_shape is not None and query_shape.intersects(obs_shape):
                                obs_cost = getattr(geom_obj.s, 'cost', 0.0)
                                if obs_cost > 0:
                                    sample_cost += obs_cost
                            continue

                        # Non-traversable obstacle
                        obs_shape = self.get_buffered_shape(geom_obj=geom_obj)
                        if obs_shape is not None and query_shape.intersects(obs_shape):
                            collision_info["blocked_geoms"].append(geom_name)
                            return {"traversable": False, "cost": float('inf'),
                                    "collision_info": collision_info,
                                    "cells_visited": cells_visited}

            # Weight the sample cost by the step distance
            # Use 1.0 as baseline if sample_cost is 0 (free space has unit cost)
            effective_cost = sample_cost if sample_cost > 0 else 1.0
            total_cost += effective_cost * step_dist

        return {"traversable": True, "cost": total_cost,
                "collision_info": collision_info, "cells_visited": cells_visited}


# --- Path validation ---    
    def validate_path(self, path, geom=None, use_shape_collision=False):
        """
        Validate that all segments are free of collisions.
        """
        if len(path) < 2:
            return False, None, {"error": "Path too short"}

        if geom is not None:
            # Check shape collision at waypoints
            for i, (x, y) in enumerate(path):
                free, info = self.check_shape_collision(x, y, geom)
                if not free:
                    return False, i, {"error": f"Shape collision at waypoint {i}", "info": info}

            # If swept-collision requested, check shape along segments
            if use_shape_collision:
                for i in range(len(path) - 1):
                    x1, y1 = path[i]
                    x2, y2 = path[i + 1]
                    free, info = self.check_segment_collision(x1, y1, x2, y2, geom=geom)
                    if not free:
                        return False, i, {"error": f"Shape collision along segment {i}", "info": info}

            return True, None, {}

        else:
            for i in range(len(path) - 1):
                (x1, y1), (x2, y2) = path[i], path[i+1]

                free, coll = self.check_segment_collision(x1, y1, x2, y2)
                if not free:
                    return False, i, {"invalid_segment": i, "collision_points": coll}

        return True, None, {}


# --- Cost computation ---
    ## TODO: need to fix this to use calc_metric
    def compute_path_cost(self, path, geom=None, shape_name="shape", cost_fn=None):
        """
        Compute path cost.

        If geom is None (point agent): uses the existing trapezoidal-average
        point-query model (avg endpoint cost × segment length).

        If geom is provided (shape agent): sweeps the shape along each segment,
        accumulating the cost of all cells/obstacles the agent overlaps.

        Parameters
        ----------
        path : list of (x, y) tuples
            The path waypoints.
        geom : GeomPoint/GeomPoly/GeomLine, optional
            The agent shape. If None, treat agent as a point.
        shape_name : str
            Name for get_buffered_shape lookup.
        cost_fn : callable, optional
            User-supplied cost function override.

        Returns
        -------
        float
            Total path cost. Returns inf if path is infeasible.
        """
        if len(path) < 2:
            return float('inf')

        segment_costs = []

        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            if geom is None:
                is_free, _ = self.check_segment_collision(x1, y1, x2, y2)
                if not is_free:
                    return float('inf')

                pt1 = self.query_point(x1, y1)
                pt2 = self.query_point(x2, y2)

                if not pt1["traversable"] or not pt2["traversable"]:
                    return float('inf')

                c1 = pt1["cost"]
                c2 = pt2["cost"]

                if c1 == float('inf') or c2 == float('inf'):
                    return float('inf')

                # Use average cost; default to 1.0 if cost is 0 (free space baseline)
                avg = 0.5 * ((c1 if c1 > 0 else 1.0) + (c2 if c2 > 0 else 1.0))
                segdist = math.hypot(x2 - x1, y2 - y1)
                segment_costs.append(avg * segdist)

            else:
                result = self._query_segment_shape(x1, y1, x2, y2, geom,
                                                shape_name=shape_name)
                if not result["traversable"]:
                    return float('inf')

                segment_costs.append(result["cost"])

        # Apply custom cost function if provided
        if cost_fn is not None:
            self.cost_function = cost_fn

        if self.cost_function is not None:
            fn = self.cost_function
            try:
                return float(fn(path, self))
            except TypeError:
                pass
            try:
                return float(fn(path))
            except TypeError:
                pass
            try:
                return float(fn(segment_costs))
            except Exception as e:
                print(f"Warning: custom cost function failed: {e}")

        # Default: sum of segment costs
        return sum(segment_costs)

    
# --- Planning ---
    def compute_path(self, start, goal, planner=None, geom=None, **kwargs):
        """
        Use external planner to compute a path.
        """
        
        goal_check = self.check_goal_feasible(goal, geom=geom)
        if not goal_check["feasible"]:
            self.replan_reasons.append(goal_check["reason"])
            self.s.flightplan = ()
            self.s.planned = False
            return ()

        planner = planner
        if planner is None:
            raise RuntimeError("No planner supplied.")

        path = planner(start, goal, self, **kwargs)

        self.computation_history.append({"start": start,
                                         "goal": goal,
                                         "path": path,
                                         "cost": self.compute_path_cost(path) if path else None,})

        self.s.flightplan = tuple(path) if path else ()
        self.s.pt = 0
        self.s.planned = bool(path)

        return self.s.flightplan

    def plan_and_validate(self, start, goal, planner=None, geom=None, use_shape_collision=False, **kwargs):
        """
        Plan, validate, and if needed replan.
        """
        attempts = 0
        max_attempts = getattr(self.p, "max_replan_attempts", 3)
        self.s.last_valid_path = ()

        while attempts < max_attempts:
            path = self.compute_path(start, goal, planner, geom=geom, **kwargs)

            if not path:
                attempts += 1
                self.replan_reasons.append("Planner returned empty path.")
                continue

            valid, bad_idx, details = self.validate_path(path, geom=geom, use_shape_collision=use_shape_collision)
            if valid:
                self.s.last_valid_path = tuple(path)
                self.s.replanning_triggered = False
                return path, attempts

            self.rejected_paths.append({"path": path,
                                        "invalid_segment": bad_idx,
                                        "details": details,
                                        "attempt": attempts,})

            self.replan_reasons.append(f"Invalid segment {bad_idx}")
            attempts += 1

        self.s.replanning_triggered = True
        return (), attempts
        #return getattr(self.s, "last_valid_path", ()), attempts


# --- Stepping / movement ---
    def next_position(self):
        """
        Return the next point in flightplan.

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.s.flightplan = ((0, 0), (1, 1), (2, 2))
        >>> pp.s.planned = True
        >>> pp.s.pt = 0
        >>> pp.next_position()
        (0, 0)
        >>> pp.next_position()
        (1, 1)
        >>> pp.next_position()
        (2, 2)
        >>> pp.next_position() is None
        True
        """
        if not self.s.planned or not self.s.flightplan:
            return None

        if self.s.pt >= len(self.s.flightplan):
            return None

        wp = self.s.flightplan[self.s.pt]
        self.s.pt += 1
        return wp

    def compute_path_length(self, path):
        """
        Compute the total length of a path.
        
        Parameters
        ----------
        path : sequence of (x, y) tuples
            The path points
        
        Returns
        -------
        float
            Total length of the path (sum of segment distances)

        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> round(pp.compute_path_length([(0,0), (3,4)]),2)
        5.0
        >>> pp.compute_path_length([(0,0)])
        0.0
        >>> round(pp.compute_path_length([(0,0), (0,5), (5,5)]), 2)
        10.0
        """
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            total_length += math.hypot(dx, dy)
        
        return total_length
    
    def compute_number_path_steps(self, path):
        """
        Computes number of waypoints in a path.
        
        Examples
        --------
        >>> pp = PathPlannerBase()
        >>> pp.compute_number_path_steps([(0,0), (1,1), (2,2)])
        3
        >>> pp.compute_number_path_steps([])
        0
        """
        return len(path)




#################################################################################
# --- Dummy environment for testing ---

# Minimal State, Parameter for doc-test obstacles
class ExampleObstacleState(State):
    cost: float = 2.0
    traversable: bool = False
    goal_allowed: bool = False

class ExampleObstacleParam(GeomParameter):
    coordinates: tuple = (8.0, 8.0)   # Circle centered on (0,0)
    buffer_around: float = 0.5        # Small radius

class ExampleGeomPoint(GeomPoint):
    container_p = ExampleObstacleParam
    container_s = ExampleObstacleState

class ExampleGeomArch(GeomArchitecture):
    def __init__(self):
        self.geoms = {}
        # Add one small circle at (8,8)
        self.geoms["obstacle"] = ExampleGeomPoint()

# Minimal ExampleGrid for grid queries
class ExampleCoordsParam(CoordsParam):
    x_size: int = 10
    y_size: int = 10
    blocksize: float = 1.0

class ExampleGrid(Coords):
    container_p = ExampleCoordsParam
    feature_traversable = (bool, True)     
    feature_goal_allowed = (bool, True)
    feature_cost = (float, 1.0)
    def init_properties(self, **kwargs):
        self.set_range("traversable", False, xmin=0, xmax=5, ymin=0, ymax=5, inclusive=False)
        self.set_range("goal_allowed", False, xmin=0, xmax=5, ymin=0, ymax=5, inclusive=False)
        self.set_range("cost", float("inf"), xmin=0, xmax=5, ymin=0, ymax=5, inclusive=False)


# Hybrid
class ExampleHybrid:
    def __init__(self):
        self.grid = ExampleGrid()
        self.geom_arch = ExampleGeomArch()
        self.geoms = self.geom_arch.geoms

if __name__ == "__main__":
    import doctest
    doctest.testmod()
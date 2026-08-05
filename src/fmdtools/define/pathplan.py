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


## NOTE: block_size name may be confusing with CoordsParam blocksize
class PathPlannerParameter(Parameter):
    """
    Static planner parameters (set at initialization).
    """
    max_distance: float = 5.0
    block_size: float = 2.5
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

    __slots__ = ('_env', 'cost_function',
                 'computation_history', 'rejected_paths', 'replan_reasons',)

    # Container references
    container_s = PathPlannerState   # PathPlannerState
    container_p = PathPlannerParameter   # PathPlannerParameter
    #container_m = None   # Mode

    def __init__(self, cost_function=None, **kwargs):
        super().__init__(**kwargs)
        self._env = None
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

        self._env = env

    @property
    def env(self):
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call init_environment() first.")
        return self._env


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
        return isinstance(self._env, Coords)

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
        return isinstance(self._env, GeomArchitecture)
    
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
        return (hasattr(self._env, 'grid') and 
            hasattr(self._env, 'geom_arch') and 
            isinstance(self._env.grid, Coords) and 
            isinstance(self._env.geom_arch, GeomArchitecture))

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
        >>> pp._get_geom_arch() is pp._env
        True
        >>> pp.init_environment(ExampleHybrid())
        >>> isinstance(pp._get_geom_arch(), ExampleGeomArch)
        True
        >>> pp.init_environment(ExampleGrid())
        >>> pp._get_geom_arch() is None
        True
        """
        if self.is_geom_arch():
            return self._env
        elif self.is_hybrid():
            return self._env.geom_arch
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
        grid_env = self._env.grid if self.is_hybrid() else self._env
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
    

    def check_collision(self, x, y, shape=None):
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
        if shape is None or self.p.agent_radius == 0.0:
            return self.query_point(x, y)["traversable"]
        else:
            return self.check_shape_collision(x, y, shape)[0]


    def check_goal_feasible(self, goal, shape=None):
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
        
        if shape is not None:
            # Use shape-aware collision
            is_free, collision_info = self.check_shape_collision(x, y, shape)
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
    def check_segment_collision(self, x1, y1, x2, y2):
        """
        Check if a line segment from (x1, y1) to (x2, y2) is collision-free.
        """
        collision_points = []

        # For Coords (grid) environments
        if self.is_coords() or self.is_hybrid():
            grid_env = self._env.grid if self.is_hybrid() else self._env
            try:
                grid_x1, grid_y1 = grid_env.to_index(x1, y1)
                grid_x2, grid_y2 = grid_env.to_index(x2, y2)
            except:
                # Either endpoint is outside the grid's bounds -- treat as blocked/non-traversable
                collision_points.append((x1, y1))
                collision_points.append((x2, y2))
                return False, collision_points

            cell_size = getattr(grid_env.p, 'block_size', 1.0)
            
            # Get all cells along the line using Bresenham
            if cell_size < 1.0:
                cells = self._dda_line(x1, y1, x2, y2, grid_env)
            else:
                cells = self._bresenham_line(grid_x1, grid_y1, grid_x2, grid_y2)
            
            # Check each cell for traversability
            for cell_coords in cells:
                if cell_size < 1.0:
                    gx, gy = grid_env.to_index(cell_coords[0], cell_coords[1])
                    pt = grid_env.grid[gx, gy]
                else:
                    pt = grid_env.grid[cell_coords[0], cell_coords[1]]
                    
                pt_info = self.query_point(*pt)
                
                if not pt_info["traversable"]:
                    collision_points.append(tuple(pt))
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
                    
                    shape = None
                    try:
                        if hasattr(geom_obj, 'create_shape'):
                            buffer_attrs = geom_obj.p.get_pref_attrs('buffer')
                            if buffer_attrs:
                                buffer_name = list(buffer_attrs.keys())[0]
                                shape = geom_obj.create_shape(buffer_name)
                            else:
                                shape = geom_obj.create_shape()
                            # -----------------------------
                    except Exception:
                        pass
                    
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
            Line endpoints in normalized grid coordinates (already scaled by cell size)
        
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
            grid_env = self._env.grid if self.is_hybrid() else self._env
            
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
                    try:
                        if shape.intersects(geom_obj.geom.shapely):
                            if not trav:
                                traversable = False
                            total_cost += cost
                            if not goal:
                                goal_allowed = False
                    except Exception:
                        pass

        return {"traversable": traversable,
                "cost": total_cost if traversable else float('inf'),
                "goal_allowed": goal_allowed,
                "blocked_cells": blocked_cells}

    def check_shape_collision(self, x, y, shape):
        """Check if shape at (x,y) collides with obstacles."""
        if shape is None or self.p.agent_radius == 0.0:
            return self.query_point(x, y)["traversable"], {}
        
        collision_info = {"blocked_cells": [], "blocked_geoms": []}
        
        # Translate shape to query position
        try:
            query_shape = translate(shape, xoff=x, yoff=y)
        except Exception as e:
            print(f"Warning: Could not translate shape: {e}")
            return self.query_point(x, y)["traversable"], {}
        
        # ===== GEOMETRY COLLISION CHECK (PRIORITY) =====
        if self.is_geom_arch() or self.is_hybrid():
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
                        
                        # Create shape using create_shape()
                        try:
                            # Get available buffers for this geometry
                            buffer_attrs = geom_obj.p.get_pref_attrs('buffer')
                            if buffer_attrs:
                                # Use the first available buffer
                                buffer_name = list(buffer_attrs.keys())[0]
                                obstacle_shape = geom_obj.create_shape(buffer_name)
                            else:
                                # Fall back to base shape if no buffers defined
                                obstacle_shape = geom_obj.create_shape()
                        except Exception as e:
                            continue
                        
                        if obstacle_shape is None:
                            continue
                        
                        # Check intersection
                        if query_shape.intersects(obstacle_shape):
                            collision_info["blocked_geoms"].append(geom_name)
                            return False, collision_info
            except Exception as e:
                print(f"Warning: Geometry collision check failed: {e}")
                traceback.print_exc()

        # ===== GRID COLLISION CHECK =====
        if self.is_coords() or self.is_hybrid():
            grid_env = self._env.grid if self.is_hybrid() else self._env
            blocksize = getattr(grid_env.p, 'block_size', 10.0)

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

            for gx in range(min_gx, max_gx + 1):
                for gy in range(min_gy, max_gy + 1):
                    grid_x = gx * blocksize
                    grid_y = gy * blocksize

                    cell_point = ShapelyPoint(grid_x, grid_y)
                    distance = query_shape.distance(cell_point)

                    if distance <= blocksize / 2:
                        cell_result = self.query_point(grid_x, grid_y)

                        if not cell_result["traversable"]:
                            collision_info["blocked_cells"].append((gx, gy))
                            return False, collision_info

        return True, collision_info

        '''
        # ===== GRID COLLISION CHECK =====
        if self.is_coords() or self.is_hybrid():
            try:
                grid_env = self._env.grid if self.is_hybrid() else self._env
                blocksize = getattr(grid_env.p, 'block_size', 10.0)
                
                minx, miny, maxx, maxy = query_shape.bounds
                
                min_gx = int(minx / blocksize)
                min_gy = int(miny / blocksize)
                max_gx = int(maxx / blocksize) + 1
                max_gy = int(maxy / blocksize) + 1
                
                for gx in range(min_gx, max_gx + 1):
                    for gy in range(min_gy, max_gy + 1):
                        grid_x = gx * blocksize
                        grid_y = gy * blocksize
                        
                        cell_point = ShapelyPoint(grid_x, grid_y)
                        distance = query_shape.distance(cell_point)
                        
                        if distance <= blocksize / 2:
                            cell_result = self.query_point(grid_x, grid_y)
                            
                            if not cell_result["traversable"]:
                                collision_info["blocked_cells"].append((gx, gy))
                                return False, collision_info
            except Exception as e:
                print(f"Warning: Grid collision check failed: {e}")
                traceback.print_exc()
        
        return True, collision_info
        '''


    def check_segment_collision_shape(self, x1, y1, x2, y2, shape, resolution=None):
        """
        Check if a moving shape along a segment collides with obstacles (swept volume).
        
        Uses swept volume approach: creates a buffer around the path and checks collision.
        Falls back to sampling if needed.
        
        Parameters
        ----------
        x1, y1, x2, y2 : float
            Segment endpoints
        shape : shapely.geometry
            The agent shape (e.g., Point.buffer(radius))
        resolution : float, optional
            Distance between sample points. If None, uses collision_check_resolution
        
        Returns
        -------
        tuple (is_free, collision_info)
            is_free : bool
            collision_info : dict with collision details
        """
        if resolution is None:
            resolution = getattr(self.p, 'collision_check_resolution', 0.5)
        
        # Calculate segment length
        distance = math.hypot(x2 - x1, y2 - y1)
        
        # Need at least 2 points (start and end)
        num_samples = max(2, int(distance / resolution) + 1)
        
        # Sample along the segment and check collision at each point
        # This leverages check_shape_collision which handles both grid and geom obstacles
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Check collision at this sample point
            is_free, collision_info = self.check_shape_collision(x, y, shape)
            
            if not is_free:
                return False, collision_info
        
        return True, {}

    '''
    def check_segment_collision_shape(self, x1, y1, x2, y2, shape, resolution=None):
        """
        Check if a moving shape along a segment collides with obstacles (swept volume).
        
        Uses Bresenham/DDA to get all cells in swept volume, then checks each for collision.
        Guarantees no gaps in coverage.
        
        Parameters
        ----------
        x1, y1, x2, y2 : float
            Segment endpoints
        shape : shapely.geometry
            The agent shape (e.g., Point.buffer(radius))
        resolution : float, optional
            Unused (kept for API compatibility)
        
        Returns
        -------
        tuple (is_free, collision_info)
            is_free : bool
            collision_info : dict with collision details
        """
        
        collision_info = {}
        
        # Create the path the center traces
        path = ShapelyLineString([(x1, y1), (x2, y2)])
        
        # Create the swept volume by buffering the path by the shape's extent
        swept_volume = path.buffer(self.p.agent_radius)
        
        # Get all cells that the swept volume overlaps using Bresenham/DDA
        if self.is_coords() or self.is_hybrid():
            grid_env = self._env.grid if self.is_hybrid() else self._env
            cells = self._get_swept_cells(swept_volume, grid_env)
            
            # Get center point of each cell and check collision using check_shape_collision
            for cell_coords in cells:
                cell_point = cell_coords  # Already world coordinates from _get_swept_cells
                
                # Use check_shape_collision to check this point
                # This handles both grid and geometry obstacles consistently
                is_free, collision_info = self.check_shape_collision(cell_point[0], cell_point[1], shape)
                
                if not is_free:
                    return False, collision_info
        
        # For geometry-only environments (no grid)
        if self.is_geom_arch() and not self.is_hybrid():
            geom_arch = self._get_geom_arch()
            
            if geom_arch is not None:
                for geom_name, geom_obj in geom_arch.geoms.items():
                    if not hasattr(geom_obj, 's'):
                        continue
                    
                    traversable = getattr(geom_obj.s, "traversable", True)
                    if traversable:
                        continue
                    
                    # Inside the commented-out check_segment_collision_shape
                    try:
                        buffer_attrs = geom_obj.p.get_pref_attrs('buffer')
                        if buffer_attrs:
                            buffer_name = list(buffer_attrs.keys())[0]
                            obstacle_shape = geom_obj.create_shape(buffer_name)
                        else:
                            obstacle_shape = geom_obj.create_shape()
                    except Exception:
                        continue
                    
                    if obstacle_shape is None:
                        continue
                    
                    # Check if swept volume intersects obstacle
                    if swept_volume.intersects(obstacle_shape):
                        return False, {"blocked_geoms": [geom_name]}
        
        return True, {}


    def _get_swept_cells(self, swept_volume, grid_env):
        """
        Get all grid cells that the swept volume overlaps using Bresenham/DDA.
        
        Returns cells as world coordinates (not grid indices) for consistency with check_shape_collision.
        
        Returns
        -------
        list of tuples
            Cell coordinates in world space (x, y)
        """
        cell_size = getattr(grid_env.p, 'block_size', 1.0)
        minx, miny, maxx, maxy = swept_volume.bounds
        
        if cell_size < 1.0:
            # Use DDA-style sampling for sub-unit cells
            return self._dda_swept_cells_optimized(minx, miny, maxx, maxy, swept_volume, cell_size, grid_env)
        else:
            # Use Bresenham-inspired boundary tracing + scanline fill for regular grid
            return self._bresenham_swept_cells_optimized(minx, miny, maxx, maxy, swept_volume, grid_env)


    def _bresenham_swept_cells_optimized(self, minx, miny, maxx, maxy, swept_volume, grid_env):
        """
        Optimized Bresenham-style approach: trace boundary and use scanline fill.
        
        Returns cells in world coordinates.
        """
        
        cells = set()
        min_gx, min_gy = grid_env.to_index(minx, miny)
        max_gx, max_gy = grid_env.to_index(maxx, maxy)
        
        block_size = getattr(grid_env.p, 'block_size', 1.0)
        
        # Get the exterior ring coordinates of swept volume
        if hasattr(swept_volume, 'exterior'):
            boundary_coords = list(swept_volume.exterior.coords)
        else:
            # For non-polygon geometries, use bounds with a buffer check
            return self._bounds_scanline_fill(min_gx, max_gx, min_gy, max_gy, swept_volume, grid_env)
        
        # Convert boundary coordinates to grid cells and trace them
        boundary_cells = set()
        for i in range(len(boundary_coords) - 1):
            x1, y1 = boundary_coords[i]
            x2, y2 = boundary_coords[i + 1]
            
            gx1, gy1 = grid_env.to_index(x1, y1)
            gx2, gy2 = grid_env.to_index(x2, y2)
            
            # Use Bresenham to trace the boundary segment
            segment_cells = self._bresenham_line(gx1, gy1, gx2, gy2)
            boundary_cells.update(segment_cells)
        
        # Convert grid indices to world coordinates and add boundary cells
        world_cells = set()
        for gx, gy in boundary_cells:
            if 0 <= gx < grid_env.grid.shape[0] and 0 <= gy < grid_env.grid.shape[1]:
                pt = grid_env.grid[gx, gy]
                world_cells.add(tuple(pt))
        
        # Scanline fill: for each row, find intervals and fill them
        scanline_cells = self._scanline_fill(min_gy, max_gy, min_gx, max_gx, swept_volume, grid_env)
        world_cells.update(scanline_cells)
        
        return list(world_cells)


    def _scanline_fill(self, min_gy, max_gy, min_gx, max_gx, swept_volume, grid_env):
        """
        Scanline fill algorithm: returns world coordinates of cells.
        """
        
        cells = set()
        block_size = getattr(grid_env.p, 'block_size', 1.0)
        
        # For each row (gy)
        for gy in range(min_gy, max_gy + 1):
            # Get world y-coordinate for this row
            world_y = grid_env.grid[0, gy][1] if hasattr(grid_env.grid[0, gy], '__getitem__') else gy * block_size
            
            # Find horizontal intervals that intersect the swept volume
            intervals = self._find_horizontal_intervals(world_y, min_gx, max_gx, swept_volume, grid_env)
            
            # Add all cells in the intervals (as world coordinates)
            for gx_start, gx_end in intervals:
                for gx in range(gx_start, gx_end + 1):
                    if 0 <= gx < grid_env.grid.shape[0]:
                        pt = grid_env.grid[gx, gy]
                        cells.add(tuple(pt))
        
        return cells


    def _find_horizontal_intervals(self, world_y, min_gx, max_gx, swept_volume, grid_env):
        """
        For a given y-coordinate, find which x-intervals (grid columns) intersect the swept volume.
        
        Returns grid indices, not world coordinates.
        """
        
        block_size = getattr(grid_env.p, 'block_size', 1.0)
        
        # Create a horizontal line at this y
        minx, _, maxx, _ = swept_volume.bounds
        horizontal_line = ShapelyLineString([(minx - block_size, world_y), (maxx + block_size, world_y)])
        
        # Get intersection points with swept volume
        if not swept_volume.intersects(horizontal_line):
            return []
        
        intersection = swept_volume.intersection(horizontal_line)
        
        # Extract x-coordinates from intersection
        x_coords = []
        
        if hasattr(intersection, 'geoms'):  # MultiLineString or similar
            for geom in intersection.geoms:
                if hasattr(geom, 'coords'):
                    x_coords.extend([coord[0] for coord in geom.coords])
        elif hasattr(intersection, 'coords'):  # ShapelyLineString or LinearRing
            x_coords.extend([coord[0] for coord in intersection.coords])
        elif hasattr(intersection, 'x'):  # Point
            x_coords.append(intersection.x)
        
        if not x_coords:
            return []
        
        # Sort x-coordinates
        x_coords.sort()
        
        # Convert to grid intervals (return grid indices)
        intervals = []
        for i in range(0, len(x_coords) - 1, 2):
            x_start = x_coords[i]
            x_end = x_coords[i + 1] if i + 1 < len(x_coords) else x_coords[i]
            
            gx_start, _ = grid_env.to_index(x_start, world_y)
            gx_end, _ = grid_env.to_index(x_end, world_y)
            
            intervals.append((min(gx_start, gx_end), max(gx_start, gx_end)))
        
        return intervals


    def _bounds_scanline_fill(self, min_gx, max_gx, min_gy, max_gy, swept_volume, grid_env):
        """
        Fallback scanline fill for non-polygon geometries using distance checking.
        Returns world coordinates.
        """
        
        cells = set()
        block_size = getattr(grid_env.p, 'block_size', 1.0)
        
        for gy in range(min_gy, max_gy + 1):
            for gx in range(min_gx, max_gx + 1):
                if 0 <= gx < grid_env.grid.shape[0] and 0 <= gy < grid_env.grid.shape[1]:
                    pt = grid_env.grid[gx, gy]
                    cell_point = ShapelyPoint(pt[0], pt[1])
                    
                    # Check if cell center is in swept volume or close enough to boundary
                    if swept_volume.contains(cell_point) or swept_volume.distance(cell_point) < block_size / 2:
                        cells.add(tuple(pt))
        
        return cells


    def _dda_swept_cells_optimized(self, minx, miny, maxx, maxy, swept_volume, cell_size, grid_env):
        """
        Optimized DDA approach for sub-unit cells.
        Returns world coordinates.
        """
        
        cells = set()
        
        # Phase 1: Trace the boundary more finely
        if hasattr(swept_volume, 'exterior'):
            boundary_coords = list(swept_volume.exterior.coords)
            for i in range(len(boundary_coords) - 1):
                x1, y1 = boundary_coords[i]
                x2, y2 = boundary_coords[i + 1]
                
                # DDA along boundary with fine resolution
                boundary_samples = self._dda_line(x1, y1, x2, y2)
                for x, y in boundary_samples:
                    cells.add((x, y))
                
                # Also add nearby cells for thickness
                for x, y in boundary_samples:
                    for dx in [-cell_size, 0, cell_size]:
                        for dy in [-cell_size, 0, cell_size]:
                            if swept_volume.contains(ShapelyPoint(x + dx, y + dy)):
                                cells.add((x + dx, y + dy))
        
        # Phase 2: Fill interior with coarser sampling
        x = minx + cell_size
        while x < maxx:
            y = miny + cell_size
            while y < maxy:
                point = ShapelyPoint(x, y)
                if swept_volume.contains(point):
                    cells.add((x, y))
                y += cell_size * 2  # Coarser interior sampling
            x += cell_size * 2
        
        return list(cells)
    '''

# --- Path validation ---
    def validate_path_shape(self, path, shape, use_shape_collision=False):
        """
        Validate path for a shape-based agent.
        
        Parameters
        ----------
        path : list of (x, y) tuples
            Path waypoints
        shape : shapely.geometry.base.BaseGeometry
            Agent shape (e.g., circle, polygon)
        use_shape_collision : bool
            If True, check swept collision along segments.
            If False, only check shape at waypoints.
        
        Returns
        -------
        tuple
            (is_valid: bool, bad_index: int or None, details: dict)
        """
        if len(path) < 2:
            return False, None, {"error": "Path too short"}
        
        # Check start and end positions
        for i, (x, y) in enumerate(path):
            free, info = self.check_shape_collision(x, y, shape)
            if not free:
                return False, i, {"error": f"Shape collision at waypoint {i}", "info": info}
        
        # Check segments if requested
        if use_shape_collision:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                free, info = self.check_segment_collision_shape(x1, y1, x2, y2, shape)
                if not free:
                    return False, i, {"error": f"Shape collision along segment {i}", "info": info}
        
        return True, None, {}

    
    def validate_path(self, path):
        """
        Validate that all segments are free of collisions.
        """
        if len(path) < 2:
            return False, None, {"error": "Path too short"}

        for i in range(len(path) - 1):
            (x1, y1), (x2, y2) = path[i], path[i+1]

            free, coll = self.check_segment_collision(x1, y1, x2, y2)
            if not free:
                return False, i, {"invalid_segment": i, "collision_points": coll}

        return True, None, {}


# --- Cost computation ---
    ## TODO: need to fix this to use calc_metric
    def compute_path_cost(self, path, cost_fn=None):
        """
        Compute path cost using either:
        - user-supplied cost_function
        - default segment cost model

        Returns float
        """
        if len(path) < 2:
            return float('inf')

        segment_costs = []
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Check if the segment itself is collision-free (not just endpoints)
            is_free, _ = self.check_segment_collision(x1, y1, x2, y2)
            if not is_free:
                return float('inf')
            
            # Query both points for cost
            pt1 = self.query_point(x1, y1)
            pt2 = self.query_point(x2, y2)

            if not pt1["traversable"] or not pt2["traversable"]:
                return float('inf')

            c1 = pt1["cost"]
            c2 = pt2["cost"]
            
            # Handle case where cost might be 0 or inf
            if c1 == float('inf') or c2 == float('inf'):
                return float('inf')
            
            # Use average cost, but default to 1.0 if cost is 0
            avg = 0.5 * ((c1 if c1 > 0 else 1.0) + (c2 if c2 > 0 else 1.0))

            dx = x2 - x1
            dy = y2 - y1
            segdist = math.hypot(dx, dy)

            segment_costs.append(avg * segdist)

        self.cost_function = cost_fn
        # Use custom cost function if provided
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

            # fallback
            return sum(segment_costs)

        # Default cost
        return sum(segment_costs)

# --- Planning ---
    def compute_path(self, start, goal, planner=None, **kwargs):
        """
        Use external planner to compute a path.
        """
        
        goal_check = self.check_goal_feasible(goal)
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

    def plan_and_validate(self, start, goal, planner=None, **kwargs):
        """
        Plan, validate, and if needed replan.
        """
        attempts = 0
        max_attempts = getattr(self.p, "max_replan_attempts", 3)

        while attempts < max_attempts:
            path = self.compute_path(start, goal, planner, **kwargs)

            if not path:
                attempts += 1
                self.replan_reasons.append("Planner returned empty path.")
                continue

            valid, bad_idx, details = self.validate_path(path)
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
        return getattr(self.s, "last_valid_path", ()), attempts


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
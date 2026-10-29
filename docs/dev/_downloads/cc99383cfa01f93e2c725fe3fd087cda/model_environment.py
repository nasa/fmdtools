"""
Beach environment model for the water-rescue simulation.

Defines the beach/water grid, where distressed swimmers appear, and the
"swimmer behavior" function that counts down distress and survival timers.

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
from fmdtools.define.object.coords import Coords, CoordsParam
from fmdtools.define.environment import Environment
from fmdtools.define.block.function import Function
from fmdtools.define.container.state import State
import numpy as np


class BeachMapParam(CoordsParam):
    """Parameters defining the size and layout of the beach grid."""
    x_size: int = 20                # width of grid (number of tiles)
    y_size: int = 5                 # length of grid (number of tiles)
    blocksize: float = 100          # 100 meters per tile
    base_locations: tuple = ((400.0, 0.0), (1600.0, 0.0))  # points where drones launch from
    rescue_locations: tuple = ((1000.0, 300.0),)           # points where people need rescue
    num_distress: int = 1           # how many swimmers will go into distress
    time_to_reach: float = 10.0     # expected time for responder to reach a victim
    vicinity_radius: float = 200.0  # default detection radius for check_vicinity()


class BeachBehaviorState(State):
    """Shared state used to communicate rescue alerts between functions."""
    time_to_reach: int = 30   # time budget for reaching a victim
    rescue: bool = False      # set True when a victim has been spotted (rescue alert)


class BeachMap(Coords):
    """
    Grid representation of the beach.

    Each tile carries per-point states (survival rate, timers, rescue status)
    and features marking what type of terrain it is (beach, water, base, etc.).
    """
    container_p = BeachMapParam
    container_s = BeachBehaviorState

    # states for every point in environment
    state_survival_rate = (float, 1)          # likelihood the victim survives
    state_person_to_rescue = (bool, False)    # True once a swimmer at this point is in distress
    point_person_location = (0, 0)            # location of the (first) distressed swimmer
    state_with_buoy = (bool, False)           # True after the drone drops a buoy at this point
    state_rescued = (bool, False)             # True once the responder completes the rescue
    state_distress_timer = (float, 100.0)     # countdown until the swimmer shows distress
    state_survival_timer = (float, 300.0)     # countdown until a distressed swimmer dies
    state_dead = (bool, False)                # True if the survival timer ran out

    # terrain features
    feature_beach = (bool, False)             # sandy area (y between 0 and blocksize)
    feature_water = (bool, False)             # open water (y beyond 2*blocksize)
    feature_base = (bool, False)              # drone launch points
    feature_rescue_location = (bool, False)   # designated victim locations

    def init_properties(self, *args, **kwargs):
        """Lay out the map: mark bases, victim points, beach strip, and water."""
        self.set_pts(self.p.base_locations, "base", True)
        self.set_pts(self.p.rescue_locations, "person_to_rescue", False)
        b = self.p.blocksize
        self.set_range("beach", True, ymin=0, ymax=b)   # first row of tiles is sand
        self.set_range("water", True, ymin=b * 2)       # everything past 2 tiles is water

        # record the first victim's location for easy lookup by other functions
        victim_pts = self.p.rescue_locations[:self.p.num_distress]
        if victim_pts:
            self.person_location = victim_pts[0]

    def check_vicinity(self, point, radius=None):
        """
        Check whether any distressed swimmer is within radius of point.

        Returns
        -------
        (in_range, closest_pt) : tuple
            in_range : True if the nearest active victim is within radius.
            closest_pt : location of that victim, or None if no one is in distress.
        """
        if radius is None:
            radius = self.p.vicinity_radius

        # gather only victims currently flagged as needing rescue
        active = []
        for pt in self.p.rescue_locations:
            if self.get(pt[0], pt[1], "person_to_rescue"):
                active.append(pt)
        if not active:
            return (False, None)

        # find the closest active victim and test it against the radius
        pts = np.array(active)
        dists = np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])
        i = int(np.argmin(dists))
        return (dists[i] <= radius, tuple(pts[i]))


class BeachEnvironment(Environment):
    """Environment flow wrapping the BeachMap grid, shared by all functions."""
    coords_c = BeachMap


class BeachBehavior(Function):
    """
    Simulates swimmer behavior over time.

    Counts down each swimmer's distress timer; once it hits zero the swimmer
    is flagged as needing rescue. From then on, a survival timer counts down —
    slower if a buoy has been dropped — and the swimmer dies if it reaches zero.
    """
    container_s = BeachBehaviorState
    flow_environment = BeachEnvironment

    def dynamic_behavior(self):
        env = self.environment.c

        # distress phase: count down until each swimmer starts struggling
        for pt in env.p.rescue_locations:
            rescued = env.get(pt[0], pt[1], "rescued")
            if rescued:
                continue  # skip already rescued victims
            timer = env.get(pt[0], pt[1], "distress_timer")
            print(f"timer: {timer}, person_to_rescue: {env.get(pt[0], pt[1], 'person_to_rescue')}")
            if timer > 0:
                env.set(pt[0], pt[1], "distress_timer", timer - self.t.dt)
            elif not env.get(pt[0], pt[1], "with_buoy"):
                # timer expired and no buoy yet -> flag swimmer as needing rescue
                env.set_pts([pt], "person_to_rescue", True)

        # survival phase: distressed swimmer's survival timer counts down
        victim = env.p.rescue_locations[0]
        if env.get(victim[0], victim[1], "distress_timer") <= 0 \
                and not env.get(victim[0], victim[1], "rescued") \
                and not env.get(victim[0], victim[1], "dead"):
            timer = env.get(victim[0], victim[1], "survival_timer")
            if env.get(victim[0], victim[1], "with_buoy"):
                timer -= self.t.dt * 0.1  # buoy slows the countdown 10x
            else:
                timer -= self.t.dt
            if timer <= 0:
                # survival time ran out -> victim dies
                timer = 0.0
                env.set_pts([victim], "dead", True)
            env.set(victim[0], victim[1], "survival_timer", timer)


# Colors used when plotting the map states/features in simulation visuals
sim_properties = {
    'beach':            {'color': 'sandybrown'},
    'water':            {'color': 'steelblue'},
    'person_to_rescue': {'color': 'red'},
    'base':             {'color': 'black'},
    'with_buoy':        {'color': 'yellow'},
    'rescued':          {'color': 'green'},
    'dead':             {'color': 'purple'},
}

from fmdtools_examples.airspacelib.wildfire_response.model_environment import FireEnvironment, double_size_p
from fmdtools_examples.airspacelib.base.aircraft import BaseAircraft
from fmdtools_examples.airspacelib.base.state import AircraftState
from model_environment import BeachMap, BeachEnvironment, sim_properties

from fmdtools.define.container.mode import Mode
from fmdtools.define.container.time import Time
import fmdtools.sim.propagate as prop
import numpy as np

import numpy as np

import numpy as np

# assuming shoreline runs along x-axis (y=0 is shore)
# 100ft ~ 30m, 200ft ~ 60m offshore
near_shore = 80   # 100ft offshore in your units
far_shore = 150    # 200ft offshore in your units

# beach runs along x-axis
x_min = 0
x_max = 2000

# zigzag between near and far shore bands, moving along the beach
zigzag_path = []
x_positions = np.arange(x_min, x_max + 100, 100)

for i, x in enumerate(x_positions):
    if i % 2 == 0:
        zigzag_path.append((x, near_shore))
    else:
        zigzag_path.append((x, far_shore))


class  BeachAircraftModes(Mode):
    opermodes = ("patrol", "rescue")
    mode: str = "patrol"
    


class BeachAircraftState(AircraftState):
    """State of aircraft. Retardant set at 100%."""
    victim_spotted: bool = False
    flight_path: list = zigzag_path#[(9000, 2750), (8696, 3420), (7828, 3987), (6531, 4367), (5000, 4500), (3469, 4367), (2172, 3987), (1304, 3420), (1000, 2750), (1304, 2080), (2172, 1513), (3469, 1133), (5000, 1000), (6531, 1133), (7828, 1513), (8696, 2080), (9000, 2750)]
    patrol_idx: int = 0
    rescue_idx: int = 0
    location: tuple = (0,0)



class BeachAircraft(BaseAircraft):
    """Aircraft that spots struggling swimmers and notifies responders"""

    container_m = BeachAircraftModes
    container_s = BeachAircraftState
    flow_environment = BeachEnvironment

    def init_block(self, **kwargs):
        """Set initial aircraft location to its assigned base."""
        self.s.x = self.environment.c.p.base_locations[self.p.base][0]
        self.s.y = self.environment.c.p.base_locations[self.p.base][1]

    def set_rescue_goal(self):
        """Determine path to fly to rescue person."""
        self.m.set_mode("rescue")
        self.send_rescue_alert()
        victim = getattr(self.environment.c, "person_location", self.environment.c.p.rescue_locations[0])
        self.s.assign(victim,"goal_x","goal_y")

    def fly_patrol(self):
        self.m.set_mode("patrol")
        waypoint = self.s.flight_path[self.s.patrol_idx]
        self.s.assign(waypoint, "goal_x", "goal_y")
        self.fly_to_goal()
        if self.indicate_at_goal():
            self.s.patrol_idx = (self.s.patrol_idx + 1) % len(self.s.flight_path)

    def fly_rescue(self):
        victim = getattr(self.environment.c, "person_location", self.environment.c.p.rescue_locations[0])
        self.s.assign(victim,"goal_x","goal_y")

        self.fly_to_goal()

        self.s.location = self.s.get_loc()

        if self.indicate_at_goal():
            self.environment.c.set_pts([victim], "with_buoy", True)
            self.environment.c.set_pts([victim], "person_to_rescue", False)
            self.m.set_mode("patrol")
    def send_rescue_alert(self):
        self.environment.c.s.rescue = True

    def dynamic_behavior(self):
        """
        If in patrol, the drone will check the area around it with the function check_vicinity. It returns a boolean called 
            in_range if the closest person is within viewing radius, and the point of the person needing rescue.
            
        If a person needs rescue, the drone will change its mode and plan a rescue path to start following until it reaches the
        person and can drop the buoy. Mode will be set back to patrol
            
        """
        if self.m.in_mode("patrol"):
            self.fly_patrol()
            # check_vicinity takes a (x, y) point tuple, not two positional args
            in_range, _ = self.environment.c.check_vicinity(self.s.get_loc())
            if in_range:
                self.set_rescue_goal()
        elif self.m.in_mode("rescue"):
            self.fly_rescue()
            #if point == goal then buoy = true and mode = patrol
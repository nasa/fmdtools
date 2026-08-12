"""
Patrol drone model for the water-rescue simulation.

The drone flies a zigzag patrol path along the shoreline. When it spots a
distressed swimmer within its viewing radius, it switches to rescue mode,
alerts the responder, flies to the victim, and drops a buoy before returning
to patrol.
"""
from fmdtools_examples.airspacelib.base.aircraft import BaseAircraft
from fmdtools_examples.airspacelib.base.state import AircraftState
from model_environment import BeachEnvironment
from fmdtools.define.container.mode import Mode
import numpy as np


# Patrol path construction
# The shoreline runs along the x-axis (y=0 is the shore).
# The drone sweeps between two offshore bands: 100ft ~ 30m, 200ft ~ 60m.
near_shore = 150   # inner band, 150ft offshore
far_shore = 270    # outer band, 270ft offshore

# extent of the beach along the x-axis
x_min = -50
x_max = 1900

# Build a zigzag path: alternate between the near and far bands while
# stepping 100 units along the beach each waypoint.
zigzag_path = []
x_positions = np.arange(x_min, x_max+100, 100)

for i, x in enumerate(x_positions):
    if i % 2 == 0:
        zigzag_path.append((x, near_shore))
    else:
        zigzag_path.append((x, far_shore))


class BeachAircraftModes(Mode):
    """Drone operating modes: patrolling the shore or rescuing a swimmer."""
    opermodes = ("patrol", "rescue")
    mode: str = "patrol"


class BeachAircraftState(AircraftState):
    """State of the drone: position, path progress, and rescue bookkeeping."""
    victim_spotted: bool = False   # whether a victim has been detected
    flight_path: list = zigzag_path  # patrol waypoints to cycle through
    patrol_idx: int = 0            # index of the current patrol waypoint
    rescue_idx: int = 0            # index along the rescue path
    time_arrived: float = -1.0     # sim time the drone reached the victim (-1 = not yet)
    location: tuple = (0, 0)       # current (x, y) position


class BeachAircraft(BaseAircraft):
    """Aircraft that spots struggling swimmers and notifies responders."""
    container_m = BeachAircraftModes
    container_s = BeachAircraftState
    flow_environment = BeachEnvironment

    def init_block(self, **kwargs):
        """Set initial aircraft location to its assigned base."""
        self.s.x = self.environment.c.p.base_locations[self.p.base][0]
        self.s.y = self.environment.c.p.base_locations[self.p.base][1]

    def set_rescue_goal(self):
        """Switch to rescue mode, alert the responder, and target the victim."""
        self.m.set_mode("rescue")
        self.send_rescue_alert()
        # fall back to the first configured rescue location if none recorded
        victim = getattr(self.environment.c, "person_location",
                         self.environment.c.p.rescue_locations[0])
        self.s.assign(victim, "goal_x", "goal_y")

    def fly_patrol(self):
        """Fly toward the current patrol waypoint; advance to the next on arrival."""
        self.m.set_mode("patrol")
        waypoint = self.s.flight_path[self.s.patrol_idx]
        self.s.assign(waypoint, "goal_x", "goal_y")
        self.fly_to_goal()
        if self.indicate_at_goal():
            # loop back to the start of the path after the last waypoint
            self.s.patrol_idx = (self.s.patrol_idx + 1) % len(self.s.flight_path)

    def fly_rescue(self):
        """Fly to the victim; on arrival drop a buoy and return to patrol."""
        victim = getattr(self.environment.c, "person_location",
                         self.environment.c.p.rescue_locations[0])
        self.s.assign(victim, "goal_x", "goal_y")

        self.fly_to_goal()

        self.s.location = self.s.get_loc()

        if self.indicate_at_goal():
            # record first arrival time
            if self.s.time_arrived < 0:
                self.s.time_arrived = self.t.time

            # drop the buoy: victim now floats and no longer needs the drone
            self.environment.c.set_pts([victim], "with_buoy", True)
            self.environment.c.set_pts([victim], "person_to_rescue", False)
            self.m.set_mode("patrol")

    def send_rescue_alert(self):
        """Raise the shared rescue flag so the responder knows to act."""
        self.environment.c.s.rescue = True

    def dynamic_behavior(self):
        """
        Per-timestep behavior.

        If in patrol, the drone checks the area around it with check_vicinity,
        which returns whether the closest distressed person is within viewing
        radius (and their location).

        If a person needs rescue, the drone switches to rescue mode and flies
        to the person to drop the buoy, then returns to patrol.
        """
        if self.m.in_mode("patrol"):
            self.fly_patrol()
            in_range, _ = self.environment.c.check_vicinity(self.s.get_loc())
            if in_range:
                self.set_rescue_goal()
        elif self.m.in_mode("rescue"):
            self.fly_rescue()

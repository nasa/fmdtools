from fmdtools_examples.airspacelib.wildfire_response.model_environment import FireEnvironment, double_size_p
from fmdtools_examples.airspacelib.base.aircraft import BaseAircraft
from fmdtools_examples.airspacelib.base.state import AircraftState
from model_environment import BeachMap, BeachEnvironment, sim_properties

from fmdtools.define.container.mode import Mode
from fmdtools.define.container.time import Time
import fmdtools.sim.propagate as prop


class  BeachAircraftModes(Mode):
    opermodes = ("patrol", "rescue")
    mode: str = "patrol"


class BeachAircraftState(AircraftState):
    """State of aircraft. Retardant set at 100%."""
    victim_spotted: bool = False
    flight_path: list = [(1000,1000),(9000,1000),(9000,4500),(1000,4500),(1000,1000)]
    patrol_idx: int = 0


class BeachAircraft(BaseAircraft):
    """Aircraft that spots struggling swimmers and notifies responders"""

    container_m = BeachAircraftModes
    container_s = BeachAircraftState
    flow_environment = BeachEnvironment

    def init_block(self, **kwargs):
        """Set initial aircraft location to its assigned base."""
        self.s.x = self.beachenvironment.c.p.base_locations[self.p.base][0]
        self.s.y = self.beachenvironment.c.p.base_locations[self.p.base][1]

    def set_fire_goal(self):
        """Determine fire to fly to to perform fire mitigation."""
        if [*self.fireenvironment.c.find_all_prop("burning")]:
            self.m.set_mode("fly_to_fire")
            pt = self.s.get("x", "y")
            closest = self.fireenvironment.c.find_closest_edge(*pt)
            if len(closest) > 0:
                self.s.assign(closest, "goal_x", "goal_y")

    def fly_patrol(self):
        self.m.set_mode("patrol")
        waypoint = self.s.flight_path[self.s.patrol_idx]
        self.s.assign(waypoint, "goal_x", "goal_y")
        self.fly_to_goal()
        if self.indicate_at_goal():
            self.s.patrol_idx = (self.s.patrol_idx + 1) % len(self.s.flight_path)

    def dynamic_behavior(self):
        if self.m.in_mode("patrol"):
            self.fly_patrol()
            if self.s.victim_spotted:
                self.m.set_mode("rescue")


if __name__ == "__main__":

    a = FireAircraft()
    fe = FireEnvironment(c={"p": {**double_size_p, "base_locations": ((42.0, 20.0),)}})
    fe.prop_time()
    # res, hist = prop.nominal(a)
    # hist.plot_line('s.fuel_status', 's.location_x', 's.location_y')
    # hist.plot_trajectory('s.location_x', 's.location_y')

    a1 = FireAircraft(s={'goal_x': 30, 'goal_y': 40}, fireenvironment=fe, track="all")

    res, hist = prop.nominal(a1, protect=False)
    hist.plot_line('s.fuel_status', 's.x', 's.y', 'm.mode')

    fig, ax = a1.fireenvironment.c.show_from(55, hist.fireenvironment.c,
                                             properties={'burning': {"color": "red", "as_bool": True}, "base": {"color": "grey"}, "extinguished": {"color": "blue", "alpha": 0.5}})
    fig, ax = a1.fireenvironment.c.show_base_placement(fig, ax)
    hist.plot_trajectory('s.x', 's.y', fig=fig, ax=ax, mark_time=True, time_ticks=2.0)
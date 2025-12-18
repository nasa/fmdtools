# -*- coding: utf-8 -*-
"""
Module for assets (e.g. aircraft, etc.) that fight the fire.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from examples.airspacelib.wildfireresponse.environment import FireEnvironment, double_size_p
from examples.airspacelib.base.aircraft import BaseAircraft
from examples.airspacelib.base.state import AircraftState

from fmdtools.define.container.mode import Mode
from fmdtools.define.container.time import Time
import fmdtools.sim.propagate as prop


class  AircraftModes(Mode):
    """
    Aircraft modes defining firefighing behavior.

    Modes
    -----
    resupply: Mode
        Resupplying fuel and retardant at air base.
    fly_to_fire : Mode
        Flying to the fire.
    mitigate_fire : Mode
        Performing a fire mitigation action (i.e., water drop).
    fly_to_base : Mode
        Flying to the base for resupply.
    """

    opermodes = ("resupply", "fly_to_fire", "mitigate_fire", "fly_to_base")
    mode: str = "resupply"


class AircraftTime(Time):
    """Timers for aircraft resupply."""

    timernames = ('resupply', )


class FireAircraftState(AircraftState):
    """State of aircraft. Retardant set at 100%."""

    retardant_status: float = 100  # starting retardant at 100%


class FireAircraft(BaseAircraft):
    """Aircraft that mitigates fire propagation and resupplies from bases."""

    container_m = AircraftModes
    container_t = AircraftTime
    container_s = FireAircraftState
    flow_fireenvironment = FireEnvironment

    def init_block(self, **kwargs):
        """Set initial aircraft location to its assigned base."""
        self.s.x = self.fireenvironment.c.p.base_locations[self.p.base][0]
        self.s.y = self.fireenvironment.c.p.base_locations[self.p.base][1]

    def set_fire_goal(self):
        """Determine fire to fly to to perform fire mitigation."""
        if [*self.fireenvironment.c.find_all_prop("burning")]:
            self.m.set_mode("fly_to_fire")
            pt = self.s.get("x", "y")
            closest = self.fireenvironment.c.find_closest_edge(*pt)
            if len(closest) > 0:
                self.s.assign(closest, "goal_x", "goal_y")

    def dynamic_behavior(self):
        """Overall dynamic behavior of the Aircraft."""
        if self.m.in_mode("resupply"):
            if self.t.timers['resupply'].indicate_complete() or self.t.timers['resupply'].indicate_standby():
                self.s.retardant_status = 100
                self.s.fuel_status = 100
                self.t.timers['resupply'].set_timer(self.p.resupply_time)
                self.m.set_mode('fly_to_fire')
                self.set_fire_goal()
            else:
                self.t.timers['resupply'].inc()
        elif self.m.in_mode("fly_to_fire"):
            self.set_fire_goal()
            self.fly_to_goal()
            if self.indicate_at_goal():
                self.m.set_mode("mitigate_fire")
        elif self.m.in_mode("mitigate_fire"):
            self.s.retardant_status = 0
            loc = self.s.gett("x", "y")
            if not self.fireenvironment.c.get(*loc, "base"):
                self.fireenvironment.c.set(*loc, "extinguished", True)
            self.fireenvironment.c.set(*loc, "burning", False)
            self.m.set_mode("fly_to_base")
            self.s.goal_x = self.fireenvironment.c.p.base_locations[self.p.base][0]
            self.s.goal_y = self.fireenvironment.c.p.base_locations[self.p.base][1]
        elif self.m.in_mode("fly_to_base"):
            self.fly_to_goal()
            if self.indicate_at_goal():
                self.m.set_mode("resupply")


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
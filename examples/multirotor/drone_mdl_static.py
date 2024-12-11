#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multirotor drone model (base model).

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

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.flow.base import Flow

import numpy as np

# MODEL FLOWS

class EEState(State):
    """
    State of electrical energy.

    Fields
    -------
    rate : float
        Dimensionless current. 1.0 is nominal.
    effort : float
        Dimensionless voltage. 1.0 is nominal while 0.0 is off.
    """

    rate: float = 1.0
    effort: float = 1.0


class EE(Flow):
    """Electrical Energy Flow."""

    __slots__ = ()
    container_s = EEState


class ForceState(State):
    """State of force. Holds support field."""

    support: float = 1.0


class Force(Flow):
    """Force flow."""

    __slots__ = ()
    container_s = ForceState


class ControlState(State):
    """
    State of control power/throttle controlling velocity.

    Fields
    -------
    forward : float
        Throttle forward. 1.0 is normal speed. 0.0 is stopped.
    upward : float
        Throttle upward. 1.0 maintains hover. 2.0 climbs and 0.0 is no throttle.
    """

    forward: float = 1.0
    upward: float = 1.0


class Control(Flow):
    """Control Flow."""

    __slots__ = ()
    container_s = ControlState


class DOFstate(State):
    """
    State defining the (simplified) degrees of freedom of the drone.

    Fields
    -------
    vertvel : float
        Vertical velocity (m/min)
    planvel : float
        Planar velocity (m/min)
    planpwr : float
        Planar power. 1.0 is normal speed and 0.0 is stopped.
    uppwr : float
        Upward power. 1.0 maintains hover. 2.0 climbs and 0.0 is no power.
    x : float
        x-position
    y : float
        y-position
    z : float
        z-position
    """

    vertvel: float = 1.0
    planvel: float = 1.0
    planpwr: float = 1.0
    uppwr: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class DOFParam(Parameter):
    """Parameter defining max velocity (5m/s or 300 m/min)."""

    max_vel: float = 300.0


class DOFs(Flow):
    """Flow defining the Drone degrees of freedom."""

    __slots__ = ()
    container_s = DOFstate
    container_p = DOFParam


class DesTrajState(State):
    """
    State defining the drone's desired trajectory.

    Fields
    -------
    dx : float
        x-distance to goal
    dy : float
        y-distance to goal
    dz : float
        z-distance to goal
    power : float
        whether to follow trajectory (0.0 no, 1.0 yes).
    """

    dx: float = 1.0
    dy: float = 0.0
    dz: float = 0.0
    power: float = 1.0

    def unit_vect2d(self):
        """
        Produce a unit vector corresponding to the planar (x and y) trajectory.

        Returns
        -------
        uvect : np.array
            Unit vector of the direction

        e.g.::
        >>> d = DesTrajState()
        >>> d.unit_vect2d()
        array([1., 0.])
        """
        return np.round(np.array([self.dx, self.dy])/self.dist2d())

    def dist2d(self):
        """
        Get the planar (x and y) distance to the goal.

        Returns
        -------
        dist : float
            2d distance to goal.

        e.g.::
        >>> d = DesTrajState()
        >>> d.dist2d()
        1.0
        """
        dist = np.sqrt(self.dx**2 + self.dy**2)
        if dist == 0.0:
            return 0.00000001
        else:
            return dist


class DesTraj(Flow):
    """Desired trajectory flow."""

    __slots__ = ()
    container_s = DesTrajState


# MODEL FUNCTIONS

class StoreEEMode(Mode):
    """Specifies fault modes for battery (e.g., lowcharge)."""

    failrate = 1e-5
    fm_args = {"nocharge": (1, 300)}


class StoreEEState(State):
    """Battery state of charge percentage."""

    soc: float = 100.0


class StoreEE(Function):
    """Class for the battery architecture/energy storage."""

    __slots__ = ("ee_out", "fs")
    container_s = StoreEEState
    container_m = StoreEEMode
    flow_ee_out = EE
    flow_fs = Force
    flownames = {"ee_1": "ee_out", "force_st": "fs"}

    def static_behavior(self, time):
        """Source loses voltage in nocharge mode."""
        if self.m.has_fault("nocharge"):
            self.ee_out.s.effort = 0.0
        else:
            self.ee_out.s.effort = 1.0


class DistEEMode(Mode):
    """
    Power Distribution Fault modes.

    Modes
    -------
    short: Fault
        EE effort goes to zero, while rate increases to 10 (or some high value)
    degr: Fault
        Less ability to transfer EE effort
    break: Fault
        Open circuit caused by mechanical breakage, inability to tranfer EE
    """

    failrate = 1e-5
    fm_args = {"short": (0.3, 3000),
               "degr": (0.5, 1000),
               "break": (0.2, 2000)}


class DistEEState(State):
    """
    State of power distribution.

    Fields
    -------
    ee_tr: float
        Ability to transfer EE rate (current, with a nominal value of 1.0)
    ee_te: float
        Ability to transfer EE effort (voltage, with a nominal value of 1.0)
    """

    ee_tr: float = 1.0
    ee_te: float = 1.0


class DistEE(Function):
    """
    Power distribution for the drone.

    Takes in power from the battery and distributes it to the motors and control
    systems/avionics.
    """

    __slots__ = ("ee_in", "ee_mot", "ee_ctl", "st")
    container_s = DistEEState
    container_m = DistEEMode
    flow_ee_in = EE
    flow_ee_mot = EE
    flow_ee_ctl = EE
    flow_st = Force
    flownames = {"ee_1": "ee_in", "force_st": "st"}

    def set_faults(self):
        """Add faults if current is too high or support is broken."""
        if self.st.s.support < 0.5 or max(self.ee_mot.s.rate, self.ee_ctl.s.rate) > 2:
            self.m.add_fault("break")
        if self.ee_in.s.rate > 2:
            self.m.add_fault("short")

    def static_behavior(self, time):
        """
        Power distribution behavior.

        Change transference in fault modes and send EE out based on input/transferrence.
        e.g., when nominal, high effort gets passed to outgoing EE flows::
        >>> d = DistEE()
        >>> d.ee_in.s.effort = 2.0
        >>> d.static_behavior(1.0)
        >>> d.ee_mot
        ee EE flow: EEState(rate=1.0, effort=2.0)

        while fault modes modify this relationship::
        >>> d = DistEE()
        >>> d.m.add_fault("short")
        >>> d.static_behavior(1.0)
        >>> d.ee_mot
        ee EE flow: EEState(rate=1.0, effort=0.0)
        >>> d.ee_in
        ee EE flow: EEState(rate=10.0, effort=1.0)
        """
        self.set_faults()
        if self.m.has_fault("short"):
            self.s.put(ee_tr=10.0, ee_te=0.0)
        elif self.m.has_fault("break"):
            self.s.put(ee_tr=0.0, ee_te=0.0)
        elif self.m.has_fault("degr"):
            self.s.put(ee_te=0.5)
        self.ee_mot.s.effort = self.s.ee_te * self.ee_in.s.effort
        self.ee_ctl.s.effort = self.s.ee_te * self.ee_in.s.effort
        self.ee_in.s.rate = m2to1([self.ee_in.s.effort, self.s.ee_tr,
                                   0.9 * self.ee_mot.s.rate + 0.1 * self.ee_ctl.s.rate])


class HoldPayloadMode(Mode):
    """
    Multirotor structure fault modes.

    Modes
    -------
    break: Fault
        provides no support to the body and lines
    deform: Fault
        support is less than desired
    """

    failrate = 1e-6
    fm_args = {"break": (0.2, 10000),
               "deform": (0.8, 10000)}


class HoldPayloadState(State):
    """
    Landing Gear States.

    Fields
    -------
    force_gr: float
        Force from the ground
    """

    force_gr: float = 1.0


class HoldPayload(Function):
    """Drone landing gear."""

    __slots__ = ('dofs', 'force_st', 'force_lin')
    container_m = HoldPayloadMode
    container_s = HoldPayloadState
    flow_dofs = DOFs
    flow_force_st = Force
    flow_force_lin = Force

    def at_ground(self):
        """Call to check if the drone is at ground level (modified in subclasses)."""
        return self.dofs.s.z <= 0.0

    def calc_force_gr(self):
        """Calculate ground force if at ground (modified in subclasses)."""
        if self.at_ground():
            self.s.force_gr = -1.0
        else:
            self.s.force_gr = 0.0

    def static_behavior(self, time):
        """
        Ground support behavior.

        If at the ground assume crash and add a break. Then assign force/support to
        lines (force_lin) and control units (force_st).

        e.g., in the nominal case::
        >>> h = HoldPayload()
        >>> h.dofs.s.z = 1.0
        >>> h.static_behavior(1.0)
        >>> h.force_st.s
        ForceState(support=1.0)

        Or, in the drone has fallen::
        >>> h.dofs.s.z = 0.0
        >>> h.static_behavior(2.0)
        >>> h.m.faults
        {'break'}
        >>> h.force_st.s
        ForceState(support=0.0)
        """
        self.calc_force_gr()

        if self.s.force_gr < -0.8:
            self.m.add_fault('break')
        elif self.s.force_gr < -0.6:
            self.m.add_fault('deform')

        # need to transfer FG to FA & FS???
        if self.m.has_fault('break'):
            self.force_st.s.support = 0.0
        elif self.m.has_fault('deform'):
            self.force_st.s.support = 0.5
        else:
            self.force_st.s.support = 1.0
        self.force_lin.s.assign(self.force_st.s, 'support')


class AffectDOFState(State):
    """
    Behavior-affecting states for drone rotors/propellers.

    Fields
    -------
    e_to: float
        Electricity transfer (out)
    e_ti: float
        Electricity pull (in)
    ct: float
        Control transferrence
    mt: float
        Mechanical support
    pt: float
        Physical tranferrence (ability of rotor to spin)
    """

    e_to: float = 1.0
    e_ti: float = 1.0
    ct: float = 1.0
    mt: float = 1.0
    pt: float = 1.0


class AffectDOFMode(Mode):
    """
    Fault modes for drone rotors/lines.

    Modes
    -------
    short : Fault
        High EE rate in, no/low power output.
    openc : Fault
        No EE rate in, no power output.
    ctlup : Fault
        Too much power output (should fly up).
    ctldn : Fault
        Too little power output (should fly down).
    ctlbreak : Fault
        No power output. (should fall).
    mechbreak : Fault
        Rotor(s) break. Loss of power output (should fall).
    mechfriction : Fault
        Rotors have adverse fiction. EEin should go up while drone descends.
    propwarp : Fault
        Warped rotors. Should cause flight deviations (descent?).
    propstuck : Fault
        Rotor stuck. Should cause a fall and high current.
    propbreak : Fault
        Rotor breaks. Should cause a loss of power/support.
    """

    failrate = 1e-5
    fm_args = {
        "short": (0.1, 200),
        "openc": (0.1, 200),
        "ctlup": (0.2, 500),
        "ctldn": (0.2, 500),
        "ctlbreak": (0.2, 1000),
        "mechbreak": (0.1, 500),
        "mechfriction": (0.05, 500),
        "propwarp": (0.01, 200),
        "propstuck": (0.02, 200),
        "propbreak": (0.03, 200),
    }


class BaseLine(object):
    """Base class for Lines that includes fault logic affecting states."""

    __slots__ = ()

    def calc_faults(self):
        """Modify AffectDOF states based on faults."""
        self.s.put(e_ti=1.0, e_to=1.0)
        if self.m.has_fault("short"):
            self.s.put(e_ti=10, e_to=0.0)
        elif self.m.has_fault("openc"):
            self.s.put(e_ti=0.0, e_to=0.0)
        if self.m.has_fault("ctlbreak"):
            self.s.ct = 0.0
        elif self.m.has_fault("ctldn"):
            self.s.ct = 0.5
        elif self.m.has_fault("ctlup"):
            self.s.ct = 2.0
        if self.m.has_fault("mechbreak"):
            self.s.mt = 0.0
        elif self.m.has_fault("mechfriction"):
            self.s.put(mt=0.5, e_ti=2.0)
        if self.m.has_fault("propstuck"):
            self.s.put(pt=0.0, mt=0.0, e_ti=4.0)
        elif self.m.has_fault("propbreak"):
            self.s.pt = 0.0
        elif self.m.has_fault("propwarp"):
            self.s.pt = 0.5


class AffectDOF(Function, BaseLine):
    """Drone rotors that the drone through the air."""

    __slots__ = ("ee_in", "ctl_in", "dofs", "force")
    container_s = AffectDOFState
    container_m = AffectDOFMode
    flow_ee_in = EE
    flow_ctl_in = Control
    flow_dofs = DOFs
    flow_force = Force
    flownames = {"ee_mot": "ee_in",
                 "ctl": "ctl_in",
                 "force_lin": "force"}

    def static_behavior(self, time):
        """
        Drone locomotive behaviors.

        Changes drone position, velocity and power based on control inputs and fualts.
        In this (static) version, this means maintaining drone height. e.g, in the
        nominal case ::
        >>> a = AffectDOF()
        >>> a.dofs.s.z
        0.0
        >>> a.static_behavior(0.0)
        >>> a.dofs.s.z
        1.0

        Mechanical breakages (And other faults) cause a fall::
        >>> a.m.add_fault("mechbreak")
        >>> a.static_behavior(0.0)
        >>> a.s.mt
        0.0
        >>> a.dofs.s.uppwr
        0.0
        >>> a.dofs.s.z
        0.0
        """
        self.calc_faults()
        self.calc_pwr()
        self.calc_vel()
        self.inc_pos()

    def calc_pwr(self):
        """Calculate immediate power/support from AffectDOF function."""
        pwr = self.s.mul("e_to", "e_ti", "ct", "mt", "pt")
        self.ee_in.s.rate = pwr
        self.dofs.s.uppwr = self.ctl_in.s.upward * pwr
        self.dofs.s.planpwr = self.ctl_in.s.forward * pwr

    def calc_vel(self):
        """Calculate velocity given power/support."""
        self.dofs.s.vertvel = max(min(-2 + 2 * self.dofs.s.uppwr, 2), -2)
        self.dofs.s.planvel = self.dofs.s.planpwr

    def inc_pos(self):
        """Increment Drone Position."""
        if self.dofs.s.vertvel > 1.5 or self.dofs.s.vertvel < -1:
            self.m.add_fault("mechbreak")
            self.dofs.s.z = 0.0
        else:
            self.dofs.s.z = 1.0


class CtlDOFstate(State):
    """
    Controller States.

    Fields
    -------
    cs: float
        Control signal transferrence (nominally 1.0)
    power: float
        Power sent transference (nominally 1.0)
    """

    cs: float = 1.0
    power: float = 1.0


class CtlDOFMode(Mode):
    """Controller Modes, noctl (lack of control) and degctl (degraded control)."""

    failrate = 1e-5
    fm_args = {"noctl": (0.2, 10000),
               "degctl": (0.8, 10000)}


class CtlDOF(Function):
    """Drone rotor control."""

    __slots__ = ("ee_in", "des_traj", "ctl", "dofs", "fs")
    container_s = CtlDOFstate
    container_m = CtlDOFMode
    flow_ee_in = EE
    flow_des_traj = DesTraj
    flow_ctl = Control
    flow_dofs = DOFs
    flow_fs = Force
    flownames = {"ee_ctl": "ee_in", "force_st": "fs"}

    def set_faults(self):
        """If no/reduced support (from force), lose control."""
        if self.fs.s.support < 0.5:
            self.m.add_fault("noctl")

    def static_behavior(self, time):
        """
        Translate desired trajectory into control signals.

        e.g., in the nominal case::
        >>> c = CtlDOF()
        >>> c.static_behavior(0.0)
        >>> c.ctl.s
        ControlState(forward=1.0, upward=1.0)

        and in the off-nominal case::
        >>> c.m.add_fault("noctl")
        >>> c.static_behavior(0.0)
        >>> c.ctl.s
        ControlState(forward=0.0, upward=0.0)
        """
        self.set_faults()
        self.calc_cs()
        up, forward = self.calc_throttle()
        self.update_ctl(up, forward)

    def calc_cs(self):
        """Calculate signal transferrence based on faults."""
        if self.m.has_fault("noctl"):
            self.s.cs = 0.0
        elif self.m.has_fault("degctl"):
            self.s.cs = 0.5
        else:
            self.s.cs = 1.0

    def calc_throttle(self):
        """Calculate upward and forward throttle (desired/ideal)."""
        up = 1.0 + self.des_traj.s.dz / self.dofs.p.max_vel

        if self.des_traj.s.same([0.0, 0.0], 'dx', 'dy'):
            forward = 0.0
        else:
            forward = 1.0
        return up, forward

    def update_ctl(self, up, forward):
        """Update control throttle flow."""
        self.s.power = self.ee_in.s.effort * self.s.cs * self.des_traj.s.power
        self.ctl.s.put(forward=self.s.power*forward,
                       upward=self.s.power*up)
        self.ctl.s.limit(forward=(0.0, 2.0), upward=(0.0, 2.0))


class PlanPathMode(Mode):
    """Path planning modes (no location and degraded location)."""

    failrate = 1e-5
    fm_args = {"noloc": (0.2, 10000), "degloc": (0.8, 10000)}


class PlanPath(Function):
    """Drone path planning function."""

    __slots__ = ("ee_in", "dofs", "des_traj", "fs")
    container_m = PlanPathMode
    flow_ee_in = EE
    flow_dofs = DOFs
    flow_des_traj = DesTraj
    flow_fs = Force
    flownames = {"ee_ctl": "ee_in", "force_st": "fs"}

    def set_faults(self):
        """Enter "noloc" fault if loses support or velocity too high."""
        if self.fs.s.support < 0.5:
            self.m.add_fault("noloc")
        if self.dofs.s.planvel > 1.5 or self.dofs.s.planvel < 0.5:
            self.m.add_fault("noloc")

    def static_behavior(self, t):
        """
        Path planning behavior.

        Assigns trajectory based on current point. In the static case, this is just
        going forward 1.0 in the x, e.g.::
        >>> p = PlanPath()
        >>> p.static_behavior(0.0)
        >>> p.des_traj.s
        DesTrajState(dx=1.0, dy=0.0, dz=0.0, power=1.0)

        If it loses location, navigation not provided:
        >>> p.m.add_fault("noloc")
        >>> p.static_behavior(0.0)
        >>> p.des_traj.s
        DesTrajState(dx=0.0, dy=0.0, dz=0.0, power=1.0)
        """
        self.set_faults()
        self.des_traj.s.assign([1.0, 0.0, 0.0], "dx", "dy", "dz")
        # faulty behaviors
        if self.m.has_fault("noloc"):
            self.des_traj.s.assign([0.0, 0.0, 0.0], "dx", "dy", "dz")
        elif self.m.has_fault("degloc"):
            self.des_traj.s.assign([0.0, 0.0, -1.0], "dx", "dy", "dz")
        if self.ee_in.s.effort < 0.5:
            self.des_traj.s.assign([0.0, 0.0, 0.0, 0.0], "dx", "dy", "dz", "power")


def m2to1(x):
    """
    Multiply a list of numbers which may take on the values infinity or zero.

    In deciding if num is inf or zero, the earlier values take precedence

    Parameters
    ----------
    x : list
        numbers to multiply

    Returns
    -------
    y : float
        result of multiplication
    """
    if np.size(x) > 2:
        x = [x[0], m2to1(x[1:])]
    if x[0] == np.inf:
        y = np.inf
    elif x[1] == np.inf:
        if x[0] == 0.0:
            y = 0.0
        else:
            y = np.inf
    else:
        y = x[0] * x[1]
    return y


class ViewModes(Mode):
    """Drone camera modes (no view of environment)."""

    failrate = 1e-5
    fm_args = {"poorview": (0.2, 10000)}


class ViewEnvironment(Function):
    """Drone camera placeholder."""

    __slots__ = ('dofs',)
    container_m = ViewModes
    flow_dofs = DOFs


class Drone(FunctionArchitecture):
    """Static multirotor drone model (executes in a single timestep)."""

    __slots__ = ()
    default_sp = {'end_time': 0}

    def init_architecture(self, **kwargs):
        # add flows to the model
        self.add_flow("force_st", Force)
        self.add_flow("force_lin", Force)
        self.add_flow("ee_1", EE)
        self.add_flow("ee_mot", EE)
        self.add_flow("ee_ctl", EE)
        self.add_flow("ctl", Control)
        self.add_flow("dofs", DOFs)
        self.add_flow("des_traj", DesTraj)
        # add functions to the model
        self.add_fxn("store_ee", StoreEE, "ee_1", "force_st")
        self.add_fxn("dist_ee", DistEE, "ee_1", "ee_mot", "ee_ctl", "force_st")
        self.add_fxn("affect_dof", AffectDOF, "ee_mot", "ctl", "dofs", "force_lin")
        self.add_fxn("ctl_dof", CtlDOF, "ee_ctl", "des_traj", "ctl", "dofs", "force_st")
        self.add_fxn("plan_path", PlanPath, "ee_ctl", "des_traj", "force_st", "dofs")
        self.add_fxn("hold_payload", HoldPayload, "dofs", "force_lin", "force_st")
        self.add_fxn("view_env", ViewEnvironment, "dofs")

    def find_classification(self, scen, mdlhist):
        """Calculate rate, cost, expected cost based on cost of repair information."""
        modes, modeprops = self.return_faultmodes()
        repcost = sum([c["cost"] for f, m in modeprops.items() for a, c in m.items()])

        totcost = repcost
        rate = scen.rate
        expcost = totcost * rate * 1e5
        return {"rate": rate, "cost": totcost, "expected_cost": expcost}


if __name__ == "__main__":
    from fmdtools.sim import propagate
    import doctest
    doctest.testmod(verbose=True)

    static_mdl = Drone()
    endclasses, mdlhists = propagate.single_faults(static_mdl)

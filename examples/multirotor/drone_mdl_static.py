import numpy as np

from fmdtools.define.parameter import Parameter, SimParam
from fmdtools.define.state import State
from fmdtools.define.block import FxnBlock
from fmdtools.define.mode import Mode
from fmdtools.define.model import Model
from fmdtools.define.flow import Flow


# MODEL FLOWS
class EEState(State):
    rate: float = 1.0
    effort: float = 1.0


class EE(Flow):
    _init_s = EEState


class ForceState(State):
    support: float = 1.0


class Force(Flow):
    _init_s = ForceState


class ControlState(State):
    forward: float = 1.0
    upward: float = 1.0


class Control(Flow):
    _init_s = ControlState


class DOFstate(State):
    vertvel: float = 1.0
    planvel: float = 1.0
    planpwr: float = 1.0
    uppwr: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class DOFParam(Parameter):
    max_vel: float = 300.0  # 5 m/s (300 m/min)


class DOFs(Flow):
    _init_s = DOFstate
    _init_p = DOFParam


class DesTrajState(State):
    dx: float = 1.0
    dy: float = 0.0
    dz: float = 0.0
    power: float = 1.0
    def unit_vect2d(self):
        return np.array([self.dx, self.dy])/self.dist2d()

    def dist2d(self):
        return np.sqrt(self.dx**2 + self.dy**2)  + 0.00001



class DesTraj(Flow):
    _init_s = DesTrajState


# MODEL FUNCTIONS

class StoreEEMode(Mode):
    failrate = 1e-5
    faultparams = {"nocharge": (1, 300)}


class StoreEEState(State):
    soc: float = 100.0


class StoreEE(FxnBlock):
    __slots__ = ("ee_out", "fs")
    _init_s = StoreEEState
    _init_m = StoreEEMode
    _init_ee_out = EE
    _init_fs = Force
    flownames = {"ee_1": "ee_out", "force_st": "fs"}
    """
    Class for the battery architecture with:
    - StoreEEState: State
        specifies battery state of charge percentage
    - StoreEEMode: Mode
        specifies modes for battery (e.g., lowcharge)
    """

    def behavior(self, time):
        if self.m.has_fault("nocharge"):
            self.ee_out.s.effort = 0.0
        else:
            self.ee_out.s.effort = 1.0


class DistEEMode(Mode):
    failrate = 1e-5
    faultparams = {"short": (0.3, 3000),
                   "degr": (0.5, 1000),
                   "break": (0.2, 2000)}
    """
    Power Distribution Fault modes. Includes:
        - short: Fault
            EE effort goes to zero, while rate increases to 10 (or some high value)
        - degr: Fault
            Less ability to transfer EE effort
        - break: Fault
            Open circuit caused by mechanical breakage, inability to tranfer EE
    """


class DistEEState(State):
    ee_tr: float = 1.0
    ee_te: float = 1.0
    """
    State of power distribution. Has values:
        - ee_tr: float
            Ability to transfer EE rate (current, with a nominal value of 1.0)
    -   - ee_te: float
            Ability to transfer EE effort (voltage, with a nominal value of 1.0)
    """


class DistEE(FxnBlock):
    __slots__ = ("ee_in", "ee_mot", "ee_ctl", "st")
    _init_s = DistEEState
    _init_m = DistEEMode
    _init_ee_in = EE
    _init_ee_mot = EE
    _init_ee_ctl = EE
    _init_st = Force
    flownames = {"ee_1": "ee_in", "force_st": "st"}
    """
    Power distribution for the drone. Takes in power from the battery and distributes
    it to the motors and control systems/avionics.
    """

    def condfaults(self, time):
        if self.st.s.support < 0.5 or max(self.ee_mot.s.rate, self.ee_ctl.s.rate) > 2:
            self.m.add_fault("break")
        if self.ee_in.s.rate > 2:
            self.m.add_fault("short")

    def behavior(self, time):
        if self.m.has_fault("short"):
            self.s.put(ee_tr=0.0, ee_te=10.0)
        elif self.m.has_fault("break"):
            self.s.put(ee_tr=0.0, ee_te=0.0)
        elif self.m.has_fault("degr"):
            self.s.put(ee_te=0.5)
        self.ee_mot.s.effort = self.s.ee_te * self.ee_in.s.effort
        self.ee_ctl.s.effort = self.s.ee_te * self.ee_in.s.effort
        self.ee_in.s.rate = m2to1([self.ee_in.s.effort, self.s.ee_tr,
                                   0.9 * self.ee_mot.s.rate + 0.1 * self.ee_ctl.s.rate])


class HoldPayloadMode(Mode):
    failrate = 1e-6
    faultparams = {"break": (0.2, 10000),
                   "deform": (0.8, 10000)}
    """
    Multirotor structure fault modes. Includes:
        - break: Fault
            provides no support to the body and lines
        - deform: Fault
            support is less than desired
    """


class HoldPayloadState(State):
    force_gr:   float = 1.0
    """
    Landing Gear States. Has values:
        - force_gr: float
            Force from the ground
    """


class HoldPayload(FxnBlock):
    __slots__ = ('dofs', 'force_st', 'force_lin')
    _init_m = HoldPayloadMode
    _init_s = HoldPayloadState
    _init_dofs = DOFs
    _init_force_st = Force
    _init_force_lin = Force

    def at_ground(self):
        return self.dofs.s.z <= 0.0

    def dynamic_behavior(self, time):
        if self.at_ground():
            self.s.force_gr = min(-0.5, (self.dofs.s.vertvel +
                                  self.dofs.s.planvel)/self.dofs.p.max_vel)
        else:
            self.s.force_gr = 0.0
        if abs(self.s.force_gr/2) > 1.0:
            self.m.add_fault('break')
        elif abs(self.s.force_gr/2) > 0.8:
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
    e_to: float = 1.0
    e_ti: float = 1.0
    ct: float = 1.0
    mt: float = 1.0
    pt: float = 1.0
    """
    Behavior-affecting states for drone rotors/propellers. Has values:
        Eto: float
            Electricity transfer (out)
        Eti: float
            Electricity pull (in)
        Ct: float
            Control transferrence
        Mt: float
            Mechanical support
        Pt: float
            Physical tranferrence (ability of rotor to spin)
    """


class AffectDOFMode(Mode):
    failrate = 1e-5
    faultparams = {
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
    def calc_faults(self):
        """Modifies AffectDOF states based on faults"""
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
    

class AffectDOF(FxnBlock, BaseLine):  # ee_mot,ctl,dofs,force_lin HSig_dofs, RSig_dofs
    __slots__ = ("ee_in", "ctl_in", "dofs", "force")
    _init_s = AffectDOFState
    _init_m = AffectDOFMode
    _init_ee_in = EE
    _init_ctl_in = Control
    _init_dofs = DOFs
    _init_force = Force
    flownames = {"ee_mot": "ee_in",
                 "ctl": "ctl_in",
                 "force_lin": "force"}
    """
    Drone rotor architecture which moves the drone through the air based on signals
    using electrical power.
    """

    def behavior(self, time):
        self.calc_faults()
        self.calc_pwr()
        self.calc_vel()
        self.inc_pos()

    def calc_pwr(self):
        """Calculates immediate power/support from AffectDOF function"""
        self.ee_in.s.rate = self.s.e_ti
        pwr = self.s.mul("e_to", "e_ti", "ct", "mt", "pt")
        self.dofs.s.uppwr = self.ctl_in.s.upward * pwr
        self.dofs.s.planpwr = self.ctl_in.s.forward * pwr

    def calc_vel(self):
        """Calculates velocity given power/support"""
        self.dofs.s.vertvel = max(min(-2 + 2 * self.dofs.s.uppwr, 2), -2)
        self.dofs.s.planvel = self.dofs.s.planpwr

    def inc_pos(self):
        """Increments Drone Position"""
        if self.dofs.s.vertvel > 1.5 or self.dofs.s.vertvel < -1:
            self.m.add_fault("mechbreak")
            self.dofs.s.z = 0.0
        else:
            self.dofs.s.z = 1.0


class CtlDOFstate(State):
    cs: float = 1.0
    power: float = 1.0
    """
    Controller States. Has entries:
        cs: float
            Control signal transferrence (nominally 1.0)
        power: float
            Power sent transference (nominally 1.0)
    """


class CtlDOFMode(Mode):
    failrate = 1e-5
    faultparams = {"noctl": (0.2, 10000), "degctl": (0.8, 10000)}


class CtlDOF(FxnBlock):
    __slots__ = ("ee_in", "des_traj", "ctl", "dofs", "fs")
    _init_s = CtlDOFstate
    _init_m = CtlDOFMode
    _init_ee_in = EE
    _init_des_traj = DesTraj
    _init_ctl = Control
    _init_dofs = DOFs
    _init_fs = Force
    flownames = {"ee_ctl": "ee_in", "force_st": "fs"}

    def condfaults(self, time):
        if self.fs.s.support < 0.5:
            self.m.add_fault("noctl")

    def behavior(self, time):
        self.calc_cs()
        up, forward = self.calc_throttle()
        self.update_ctl(up, forward)

    def calc_cs(self):
        if self.m.has_fault("noctl"):
            self.s.cs = 0.0
        elif self.m.has_fault("degctl"):
            self.s.cs = 0.5
        else:
            self.s.cs = 1.0

    def calc_throttle(self):
        up = 1.0 + self.des_traj.s.dz / self.dofs.p.max_vel

        if self.des_traj.s.same([0.0, 0.0], 'dx', 'dy'):
            forward = 0.0
        else:
            forward = 1.0
        return up, forward

    def update_ctl(self, up, forward):
        self.s.power = self.ee_in.s.effort * self.s.cs * self.des_traj.s.power
        self.ctl.s.put(forward=self.s.power*forward,
                       upward=self.s.power*up)
        self.ctl.s.limit(forward=(0.0, 2.0), upward=(0.0, 2.0))


class PlanPathMode(Mode):
    failrate = 1e-5
    faultparams = {"noloc": (0.2, 10000), "degloc": (0.8, 10000)}


class PlanPath(FxnBlock):
    __slots__ = ("ee_in", "dofs", "des_traj", "fs")
    _init_m = PlanPathMode
    _init_ee_in = EE
    _init_dofs = DOFs
    _init_des_traj = DesTraj
    _init_fs = Force
    flownames = {"ee_ctl": "ee_in", "force_st": "fs"}

    def condfaults(self, time):
        if self.fs.s.support < 0.5:
            self.m.add_fault("noloc")
        if self.dofs.s.planvel > 1.5 or self.dofs.s.planvel < 0.5:
            self.m.add_fault("noloc")

    def behavior(self, t):
        self.des_traj.s.assign([1.0, 0.0, 0.0], "dx", "dy", "dz")
        # faulty behaviors
        if self.m.has_fault("noloc"):
            self.des_traj.s.assign([0, 0, 0], "dx", "dy", "dz")
        elif self.m.has_fault("degloc"):
            self.des_traj.s.assign([0, 0, -1], "dx", "dy", "dz")
        if self.ee_in.s.effort < 0.5:
            self.des_traj.s.assign([0.0, 0.0, 0.0, 0.0], "dx", "dy", "dz", "power")


def m2to1(x):
    """
    Multiplies a list of numbers which may take on the values infinity or zero. In deciding if num is inf or zero, the earlier values take precedence

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
    failrate = 1e-5
    faultparams = {"poorview": (0.2, 10000)}


class ViewEnvironment(FxnBlock):
    _init_m = ViewModes
    _init_dofs = DOFs


class Drone(Model):
    __slots__ = ()
    """
    Static multirotor drone model (executes in a single timestep).
    """

    def __init__(self, sp=SimParam(times=(0,)), **kwargs):
        super().__init__(sp=sp, **kwargs)
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

        self.build()

    def find_classification(self, scen, mdlhist):
        modes, modeprops = self.return_faultmodes()
        repcost = sum([c["rcost"] for f, m in modeprops.items() for a, c in m.items()])

        totcost = repcost
        rate = scen.rate
        expcost = totcost * rate * 1e5
        return {"rate": rate, "cost": totcost, "expected cost": expcost}


if __name__ == "__main__":
    from fmdtools.sim import propagate

    static_mdl = Drone()
    endclasses, mdlhists = propagate.single_faults(static_mdl)

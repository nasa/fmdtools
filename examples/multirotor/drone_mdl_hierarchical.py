# -*- coding: utf-8 -*-
"""
File name: quad_mdl.py
Author: Daniel Hulse
Created: June 2019
Description: A fault model of a multi-rotor drone.
"""
import numpy as np
import fmdtools.sim as fs
import fmdtools.analyze as an

from fmdtools.define.parameter import Parameter
from fmdtools.define.state import State
from fmdtools.define.block import Component, CompArch
from fmdtools.define.model import Model

from examples.multirotor.drone_mdl_static import m2to1, DistEE, BaseLine
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, DesTraj
from examples.multirotor.drone_mdl_static import AffectDOFMode, AffectDOFState
from examples.multirotor.drone_mdl_dynamic import StoreEE, CtlDOF, PlanPath, HoldPayload
from examples.multirotor.drone_mdl_dynamic import ViewEnvironment, DroneEnvironment
from examples.multirotor.drone_mdl_dynamic import Drone as DynDrone
from examples.multirotor.drone_mdl_dynamic import AffectDOF as AffectDOFDynamic


class OverallAffectDOFState(State):
    """
    Overall states for the multirotor AffectDOF architecture.

    Fields
    -------
    lrstab: float
        Left/Right stability (nominal value = 0.0)
    frstab: float
        Front/Rear stability (nominal value = 0.0)
    amp_factor: float
        Amplification factor (for fault recovery)
    """

    lrstab: float = 0.0
    frstab: float = 0.0
    amp_factor: float = 1.0


class AffectDOFArch(CompArch):
    """
    Line Architecture defined by parameter 'archtype'.

    Archtype has options:
    - 'quad': quadrotor architecture
    - 'hex': hexarotor architecture
    - 'oct': octorotor architecture
    Which in turn change the following fields...

    Fields
    -------
    forward : dict
        Correction factors for moving forward (based on which rotors are in front).
    upward: dict
        Correction factors for moving upward (1.0 unless in a recovery mode).
    lr_dict: dict
        Left/right dictionary. Has structure {"l": {<left components>}, ...}
    fr_dict: dict
        Front/rear dictionary. Has structure {'f':{<front components>}, ...}
    opposite:
        Component on the opposite side of a given component. Used for reconfiguration.
    """

    archtype: str = 'quad'
    forward: dict = dict()
    upward: dict = dict()
    lr_dict: dict = dict()
    fr_dict: dict = dict()
    opposite: dict = dict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.archtype == "quad":
            self.make_components(Line, 'lf', 'lr', 'rf', 'rr')
            self.forward.update({'rf': 0.5, 'lf': 0.5, 'lr': -0.5, 'rr': -0.5})
            self.lr_dict.update({'l': {'lf', 'lr'}, 'r': {'rf', 'rr'}})
            self.fr_dict.update({'f': {'lf', 'rf'}, 'r': {'lr', 'rr'}})
        elif self.archtype == "hex":
            self.make_components(Line, 'rf', 'lf', 'lr', 'rr', 'r', 'f')
            self.forward.update({'rf': 0.5, 'lf': 0.5, 'lr': -0.5,
                                'rr': -0.5, 'r': -0.75, 'f': 0.75})
            self.lr_dict.update({'l': {'lf', 'lr'}, 'r': {'rf', 'rr'}})
            self.fr_dict.update({'f': {'lf', 'rf', 'f'}, 'r': {'lr', 'rr', 'r'}})
            self.opposite.update({{'f': 'r', 'rf': 'lr', 'rr': 'lf'}})
        elif self.archtype == "oct":
            self.make_components(Line, 'lf', 'rf', 'lf2', 'rf2',
                                 'lr', 'rr', 'lr2', 'rr2')
            self.forward.update({'rf': 0.5, 'lf': 0.5, 'lr': -0.5, 'rr': -
                                0.5, 'rf2': 0.5, 'lf2': 0.5, 'lr2': -0.5, 'rr2': -0.5})
            self.lr_dict.update({'l': {'lf', 'lr', 'lf2', 'lr2'},
                                'r': {'rf', 'rr', 'rf2', 'rr2'}})
            self.fr_dict.update({'f': {'lf', 'rf', 'lf2', 'rf2'},
                                'r': {'lr', 'rr', 'lr2', 'rr2'}})
            self.opposite.update({"lf": "rr", "rf": "lr", "rf2": "lr2", "rr2": "lf2"})
        self.upward = {c: 1.0 for c in self.components}
        self.opposite.update({v: k for k, v in self.opposite.items()})


class AffectDOF(AffectDOFDynamic):
    """Rotor locomotion (multi-component extension)."""

    _init_s = OverallAffectDOFState
    _init_ca = AffectDOFArch

    def behavior(self, time):
        """Rotor dynamic behavior with architecture-base recovery."""
        if self.ca.opposite:
            self.reconfig_faults()
        self.calc_pwr()

    def reconfig_faults(self):
        """Corrects for individual line faultmodes by turning off the opposite rotor
        and upping the throttle (amp_factor)"""
        for fault in self.m.faults:
            if fault in self.ca.faultmodes:
                comp = self.ca.faultmodes[fault]
                opp = self.ca.opposite[comp]
                if self.ca.forward[comp] != 0.0:
                    self.ca.forward[comp] = 0.0
                    self.ca.upward[comp] = 0.0
                if self.ca.forward[opp] != 0.0:
                    self.ca.forward[opp] = 0.0
                    self.ca.upward[opp] = 0.0
        tot_comps = len(self.ca.components)
        empty_comps = len([c for c in self.ca.forward if self.ca.forward[c] == 0.0])
        try:
            self.s.amp_factor = tot_comps / (tot_comps - empty_comps)
        except ZeroDivisionError:
            self.s.amp_factor = 1.0

    def calc_pwr(self):
        """
        Calculates overall power and stability based on individual rotor output.

        e.g., ::
        >>> a = AffectDOF()
        >>> a.dofs.s.put(z=100.0)
        >>> a.ctl_in.s.put(forward=0.0, upward=1.0)
        >>> a.calc_pwr()
        >>> a.dofs.s
        DOFstate(vertvel=1.0, planvel=1.0, planpwr=-0.0, uppwr=1.0, x=0.0, y=0.0, z=100.0)
        >>> a.ctl_in.s.put(forward=1.0, upward=1.0)
        >>> a.calc_pwr()
        >>> a.dofs.s
        DOFstate(vertvel=1.0, planvel=1.0, planpwr=1.0, uppwr=1.0, x=0.0, y=0.0, z=100.0)
        """
        air, ee_in = {}, {}
        # injects faults into lines
        for linname, lin in self.ca.components.items():
            a, ee = lin.behavior(self.ee_in.s.effort,
                                 self.ctl_in,
                                 self.ca.upward[linname] * self.s.amp_factor,
                                 self.ca.forward[linname] * self.s.amp_factor,
                                 self.force.s.support)
            air[lin.name] = a
            ee_in[lin.name] = ee

        if any(value >= 10 for value in ee_in.values()):
            self.ee_in.s.rate = 10
        elif any(value != 0.0 for value in ee_in.values()):
            self.ee_in.s.rate = sum(ee_in.values()) / \
                len(ee_in)  # should it really be max?
        else:
            self.ee_in.s.rate = 0.0

        self.s.lrstab = (sum([air[comp] for comp in self.ca.lr_dict['l']]) -
                         sum([air[comp] for comp in self.ca.lr_dict['r']]))/len(air)
        self.s.frstab = (sum([air[comp] for comp in self.ca.fr_dict['r']]) -
                         sum([air[comp] for comp in self.ca.fr_dict['f']]))/len(air)
        if abs(self.s.lrstab) >= 0.4 or abs(self.s.frstab) >= 0.75:
            self.dofs.s.put(uppwr=0.0, planpwr=0.0)
        else:
            airs = list(air.values())
            self.dofs.s.uppwr = np.mean(airs)
            self.dofs.s.planpwr = -2*self.s.frstab


class Line(Component, BaseLine):
    """Individual version of a line (extends BaseLine in static model)."""

    _init_s = AffectDOFState
    _init_m = AffectDOFMode

    def behavior(self, ee_in, ctlin, u_fact, f_fact, force):
        """Calculate air, ee out based on inputs and modes."""
        if force <= 0.0:
            self.m.add_fault('mechbreak', 'propbreak')
        elif force <= 0.5:
            self.m.add_fault('mechfriction')

        self.calc_faults()

        airout = m2to1([ee_in,
                        self.s.e_ti,
                        ctlin.s.upward * u_fact +
                        ctlin.s.forward * f_fact,
                        self.s.ct,
                        self.s.mt,
                        self.s.pt])
        ee_in = m2to1([ee_in, self.s.e_to])
        return airout, ee_in


class DroneParam(Parameter, readonly=True):
    """Parameter defining drone architecture (quad, oct, or hex)."""

    arch: str = 'quad'
    arch_set = ('quad', 'oct', 'hex')


class Drone(DynDrone):
    """Hierarchical version of the drone model."""

    _init_p = DroneParam

    def __init__(self, **kwargs):
        Model.__init__(self, **kwargs)
        # add flows to the model
        self.add_flow('force_st', Force)
        self.add_flow('force_lin', Force)
        self.add_flow('ee_1', EE)
        self.add_flow('ee_mot', EE)
        self.add_flow('ee_ctl', EE)
        self.add_flow('ctl', Control)
        self.add_flow('dofs', DOFs)
        self.add_flow('des_traj', DesTraj)
        self.add_flow('environment', DroneEnvironment)
        # add functions to the model
        self.add_fxn('store_ee', StoreEE, 'ee_1', 'force_st')
        self.add_fxn('dist_ee', DistEE, 'ee_1', 'ee_mot', 'ee_ctl', 'force_st')
        self.add_fxn('affect_dof', AffectDOF, 'ee_mot', 'ctl', 'des_traj',
                     'dofs', 'force_lin', ca={'archtype': self.p.arch})
        self.add_fxn('ctl_dof', CtlDOF, 'ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path', PlanPath, 'ee_ctl', 'des_traj', 'force_st', 'dofs')
        self.add_fxn('hold_payload', HoldPayload, 'force_lin', 'force_st', 'dofs')
        self.add_fxn('view_env', ViewEnvironment, 'dofs', 'environment')

        self.build()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    import multiprocessing as mp
    from fmdtools.sim.sample import FaultDomain, FaultSample
    from fmdtools.analyze.phases import PhaseMap

    # check rf_mechbreack fault propagation in quad architecture:
    hierarchical_model = Drone(p=DroneParam(arch='quad'))
    endclass, mdlhist = fs.propagate.one_fault(hierarchical_model,
                                               'affect_dof',
                                               'rf_mechbreak',
                                               time=5)
    an.plot.hist(mdlhist, 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc')

    # check rr2_propstuck fault in oct architecture over several times:
    mdl = Drone(p=DroneParam(arch='oct'))
    rr2_faults = FaultDomain(mdl)
    rr2_faults.add_fault('affect_dof', 'rr2_propstuck')

    rr2_faultscens = FaultSample(rr2_faults, phasemap=PhaseMap(mdl.sp.phases))
    rr2_faultscens.add_single_fault_phases()

    endclasses, mdlhists = fs.propagate.approach(mdl,
                                                 rr2_faultscens,
                                                 staged=True,
                                                 pool=mp.Pool(4))

    # plot a single scen (at t=8)
    fault_kwargs = {'alpha': 0.2, 'color': 'red'}
    an.plot.hist(mdlhists.get('nominal', 'affect_dof_rr2_propstuck_t8p0').flatten(),
                 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc')

    # plot all scens
    an.plot.hist(mdlhists, 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc',
                 indiv_kwargs={'faulty': fault_kwargs})
    fig, ax = an.show.trajectories(mdlhists,
                                   "dofs.s.x", "dofs.s.y", "dofs.s.z",
                                   time_groups=['nominal'],
                                   indiv_kwargs={'faulty': fault_kwargs})

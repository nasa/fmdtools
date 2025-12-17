#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multirotor drone model (with component architectures).

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
from examples.multirotor.drone_mdl_static import m2to1, DistEE, BaseLine, ControlState
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, DesTraj
from examples.multirotor.drone_mdl_static import AffectDOFMode, AffectDOFState
from examples.multirotor.drone_mdl_dynamic import StoreEE, CtlDOF, PlanPath, HoldPayload
from examples.multirotor.drone_mdl_dynamic import ViewEnvironment, DroneEnvironment
from examples.multirotor.drone_mdl_dynamic import Drone as DynDrone
from examples.multirotor.drone_mdl_dynamic import AffectDOF as AffectDOFDynamic

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.block.component import Component
from fmdtools.define.architecture.component import ComponentArchitecture
from fmdtools.define.flow.multiflow import MultiFlow

import fmdtools.sim as fs

import numpy as np

class OverallAffectDOFState(AffectDOFState):
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

    lrstab: np.float64 = 0.0
    frstab: np.float64 = 0.0


class LineArchParam(Parameter):
    """
    Line Architecture defined by parameter 'archtype'.

    Archtype has options:
    - 'quad': quadrotor architecture
    - 'hex': hexarotor architecture
    - 'oct': octorotor architecture
    Which in turn change the following fields...

    Fields
    -------
    components: tuple
        Set of component names for the lines named using the convention for the str:
        0: l/r - left or right
        1: f/r - front or rear
        3: /2: - if 2, this is a secondary rotor in a similar location.
    lr_dict: dict
        Left/right dictionary. Has structure {"l": {<left components>}, ...}
    fr_dict: dict
        Front/rear dictionary. Has structure {'f':{<front components>}, ...}
    opposite:
        Component on the opposite side of a given component. Used for reconfiguration.

    Examples
    --------
    >>> LineArchParam()
    LineArchParam(archtype='quad', components=('lf', 'lr', 'rf', 'rr'), lr_dict={'l': ('lf', 'lr'), 'r': ('rf', 'rr')}, fr_dict={'f': ('lf', 'rf'), 'r': ('lr', 'rr')}, opposite={'rf': 'lr', 'rr': 'lf', 'lr': 'rf', 'lf': 'rr'})
    >>> LineArchParam(archtype='hex')
    LineArchParam(archtype='hex', components=('lf', 'lr', 'rf', 'rr', 'f', 'r'), lr_dict={'l': ('lf', 'lr'), 'r': ('rf', 'rr')}, fr_dict={'f': ('lf', 'rf', 'f'), 'r': ('lr', 'rr', 'r')}, opposite={'rf': 'lr', 'rr': 'lf', 'f': 'r', 'lr': 'rf', 'lf': 'rr', 'r': 'f'})
    >>> LineArchParam(archtype='oct')
    LineArchParam(archtype='oct', components=('lf', 'lr', 'rf', 'rr', 'lf2', 'lr2', 'rf2', 'rr2'), lr_dict={'l': ('lf', 'lr', 'lf2', 'lr2'), 'r': ('rf', 'rr', 'rf2', 'rr2')}, fr_dict={'f': ('lf', 'rf', 'lf2', 'rf2'), 'r': ('lr', 'rr', 'lr2', 'rr2')}, opposite={'rf': 'lr', 'rr': 'lf', 'rf2': 'lr2', 'rr2': 'lf2', 'lr': 'rf', 'lf': 'rr', 'lr2': 'rf2', 'lf2': 'rr2'})
    """

    archtype: str = 'quad'
    components: tuple = ()
    lr_dict: dict = dict()
    fr_dict: dict = dict()
    opposite: dict = dict()

    def __init__(self, *args, **kwargs):
        archtype = self.get_true_field('archtype', *args, **kwargs)
        if archtype == 'quad':
            components = ('lf', 'lr', 'rf', 'rr')
            lr_dict = {'l': ('lf', 'lr'), 'r': ('rf', 'rr')}
            fr_dict = {'f': ('lf', 'rf'), 'r': ('lr', 'rr')}
            opposite = {'rf': 'lr', 'rr': 'lf'}
        elif archtype == 'hex':
            components = ('lf', 'lr', 'rf', 'rr', 'f', 'r')
            lr_dict = {'l': ('lf', 'lr'), 'r': ('rf', 'rr')}
            fr_dict = {'f': ('lf', 'rf', 'f'), 'r': ('lr', 'rr', 'r')}
            opposite = {'rf': 'lr', 'rr': 'lf', 'f': 'r'}
        elif archtype == 'oct':
            components = ('lf', 'lr', 'rf', 'rr', 'lf2', 'lr2', 'rf2', 'rr2')
            lr_dict = {'l': ('lf', 'lr', 'lf2', 'lr2'), 'r': ('rf', 'rr', 'rf2', 'rr2')}
            fr_dict = {'f': ('lf', 'rf', 'lf2', 'rf2'), 'r': ('lr', 'rr', 'lr2', 'rr2')}
            opposite = {'rf': 'lr', 'rr': 'lf', 'rf2': 'lr2', 'rr2': 'lf2'}
        else:
            raise Exception("Invalid arch type")
        opposite.update({v: k for k, v in opposite.items()})
        args = self.get_true_fields(*args, archtype=archtype, components=components,
                                    lr_dict=lr_dict, fr_dict=fr_dict, opposite=opposite)
        super().__init__(*args, strict_immutability=False)

    def get_forward_factors(self):
        if self.archtype == 'quad':
            return {'rf': 0.5, 'lf': 0.5, 'lr': -0.5, 'rr': -0.5}
        elif self.archtype == 'hex':
            return {'rf': 0.5, 'lf': 0.5, 'lr': -0.5, 'rr': -0.5, 'r': -0.75, 'f': 0.75}
        elif self.archtype == 'oct':
            return {'rf': 0.5, 'lf': 0.5, 'lr': -0.5, 'rr': -0.5,
                    'rf2': 0.5, 'lf2': 0.5, 'lr2': -0.5, 'rr2': -0.5}


class MultiControl(MultiFlow):
    container_s = ControlState


class AffectDOFArch(ComponentArchitecture):
    """
    Drone line architecture (quad/hex/octorotor).

    Architecture reconfigures rotors when a fault is present, see:

    Examples
    --------
    >>> adf = AffectDOFArch()
    >>> adf.flows['ctl_in'].s.put(forward=1.0, upward=1.0)
    >>> adf.static_behavior()
    >>> adf.flows['ctl']
    ctl MultiControl
    - s=ControlState(forward=1.0, upward=1.0)
    LOCALS:
    - lf=(s=(forward=0.5, upward=1.0))
    - lf_factors=(s=(forward=0.5, upward=1.0))
    - lr=(s=(forward=-0.5, upward=1.0))
    - lr_factors=(s=(forward=-0.5, upward=1.0))
    - rf=(s=(forward=0.5, upward=1.0))
    - rf_factors=(s=(forward=0.5, upward=1.0))
    - rr=(s=(forward=-0.5, upward=1.0))
    - rr_factors=(s=(forward=-0.5, upward=1.0))
    >>> adf.comps['lf'].m.add_fault("mechbreak") # if one is broken, should adjust ctl
    >>> adf.static_behavior()
    >>> adf.flows['ctl']
    ctl MultiControl
    - s=ControlState(forward=1.0, upward=1.0)
    LOCALS:
    - lf=(s=(forward=0.0, upward=0.0))
    - lf_factors=(s=(forward=0.0, upward=0.0))
    - lr=(s=(forward=-1.0, upward=2.0))
    - lr_factors=(s=(forward=-1.0, upward=2.0))
    - rf=(s=(forward=1.0, upward=2.0))
    - rf_factors=(s=(forward=1.0, upward=2.0))
    - rr=(s=(forward=0.0, upward=0.0))
    - rr_factors=(s=(forward=0.0, upward=0.0))
    """

    container_p = LineArchParam

    def init_architecture(self, **kwargs):
        self.add_flow("ee_in", EE)
        self.add_flow("force", Force)
        self.add_flow("ctl_in", Control)
        self.add_flow("ctl", MultiControl)
        # add state configuration - relative throttle for each line
        forward = self.p.get_forward_factors()
        # add lines
        for cname in self.p.components:
            self.add_comp(cname, Line, 'ee_in', 'ctl', 'force')
            s={'forward': forward[cname], 'upward': 1.0}
            self.flows['ctl'].create_local(cname+"_factors", s=s)

    def static_behavior(self):
        """Trigger reconfiguration during static behavior step."""
        if self.p.opposite and self.get_faults(with_base_faults=False):
            self.reconfig_faults()
        self.send_local_control_signals()

    def send_local_control_signals(self):
        """Calc and send local control signals to lines."""
        for cname in self.comps:
            ctl = self.flows['ctl'].get_view(cname)
            facts = self.flows['ctl'].get_view(cname+"_factors")
            ctl.s.put(forward=facts.s.forward*self.flows['ctl_in'].s.forward,
                      upward=facts.s.upward*self.flows['ctl_in'].s.upward)

    def reconfig_faults(self):
        """
        Correct for individual line faultmodes.

        Turns off the opposite rotor and upping the throttle (amp_factor).
        """
        empty_comps = [c for c, comp in self.comps.items()
                       if comp.m.faults or self.comps[self.p.opposite[c]].m.faults]
        num_remaining = len(self.comps)-len(empty_comps)
        if num_remaining:
            amp_factor = len(self.comps)/num_remaining
        else:
            amp_factor = 0.0
        forward = self.p.get_forward_factors()
        for cname, comp in self.comps.items():
            if cname in empty_comps:
                new_facts = dict(forward=0.0, upward=0.0)
            else:
                new_facts = dict(forward=forward[cname]*amp_factor, upward=amp_factor)
            self.flows['ctl'].get_view(cname+'_factors').s.put(**new_facts)


class AffectDOF(AffectDOFDynamic):
    """
    Multirotor locomotion (multi-component extension).

    Each rotor is simulated individually, producing current and air values that are
    aggregated in the power output calculation.

    Examples
    --------
    >>> a = AffectDOF()
    >>> a.dofs.s.put(z=100.0)
    >>> a.ctl_in.s.put(forward=0.0, upward=1.0) # ascent - only going up
    >>> a.ca() # simulate component arch
    >>> a.calc_pwr()
    >>> a.dofs.s
    DOFstate(vertvel=1.0, planvel=1.0, planpwr=-0.0, uppwr=1.0, x=0.0, y=0.0, z=100.0)
    >>> a.ctl_in.s.put(forward=1.0, upward=1.0) # ascent and forward flight - both
    >>> a.ca() # simulate component arch
    >>> a.calc_pwr()
    >>> a.dofs.s
    DOFstate(vertvel=1.0, planvel=1.0, planpwr=1.0, uppwr=1.0, x=0.0, y=0.0, z=100.0)
    """

    container_s = OverallAffectDOFState
    arch_ca = AffectDOFArch

    def calc_pwr(self):
        """
        Calculate overall power and stability based on individual rotor output.

        Aggregates current and air from contained component architecture lines to
        calculate high-level output to dofs.
        """
        over_state = self.s.mul("e_to", "e_ti", "ct", "mt", "pt")
        airs = {cname: c.s.air for cname, c in self.ca.comps.items()}
        currents = {cname: c.ee.s.rate for cname, c in self.ca.comps.items()}

        if any(value >= 10 for value in currents.values()):
            self.ee_in.s.rate = 10
        elif any(value != 0.0 for value in currents.values()):
            self.ee_in.s.rate = sum(currents.values()) / len(currents)  # should it really be max?
        else:
            self.ee_in.s.rate = 0.0

        self.s.lrstab = (sum([airs[comp] for comp in self.ca.p.lr_dict['l']]) -
                         sum([airs[comp] for comp in self.ca.p.lr_dict['r']]))/len(airs)
        self.s.frstab = (sum([airs[comp] for comp in self.ca.p.fr_dict['r']]) -
                         sum([airs[comp] for comp in self.ca.p.fr_dict['f']]))/len(airs)
        if abs(self.s.lrstab) >= 0.4 or abs(self.s.frstab) >= 0.75:
            self.dofs.s.put(uppwr=0.0, planpwr=0.0)
        else:
            self.dofs.s.uppwr = np.mean(list(airs.values())) * over_state
            self.dofs.s.planpwr = -2*self.s.frstab * over_state


class LineState(AffectDOFState):
    """AffectDOFState with air output for individual rotor."""

    air : np.float64 = 0.0

class Line(Component, BaseLine):
    """Individual version of a line (extends BaseLine in static model)."""

    container_s = LineState
    container_m = AffectDOFMode
    flow_ee_in = EE
    flow_ee = EE
    flow_ctl = MultiControl
    flow_force = Force

    def init_block(self, **kwargs):
        self.ctl = self.ctl.create_local(self.name)

    def static_behavior(self):
        if self.force.s.support <= 0.0:
            self.m.add_fault('mechbreak', 'propbreak')
        elif self.force.s.support <= 0.5:
            self.m.add_fault('mechfriction')
        self.calc_faults()
        pwr = self.ctl.s.upward + self.ctl.s.forward
        self.ee.s.effort = self.ee_in.s.effort
        self.ee.s.rate = m2to1([self.ee_in.s.effort, self.s.e_to, pwr])
        self.s.air = m2to1([self.ee_in.s.effort, self.s.e_ti, pwr, self.s.ct, self.s.mt,
                            self.s.pt])


class DroneParam(Parameter, readonly=True):
    """Parameter defining drone architecture (quad, oct, or hex)."""

    arch: str = 'quad'
    arch_set = ('quad', 'oct', 'hex')


class Drone(DynDrone):
    """Hierarchical version of the drone model."""

    container_p = DroneParam

    def init_architecture(self, **kwargs):
        # add flows to the model
        self.add_flow('force_st', Force)
        self.add_flow('force_lin', Force)
        self.add_flow('ee_1', EE, s={'rate': 0.0})
        self.add_flow('ee_mot', EE, s={'rate': 0.0})
        self.add_flow('ee_ctl', EE, s={'rate': 0.0})
        self.add_flow('ctl', Control)
        self.add_flow('dofs', DOFs)
        self.add_flow('des_traj', DesTraj)
        self.add_flow('environment', DroneEnvironment)
        # add functions to the model
        self.add_fxn('store_ee', StoreEE, 'ee_1', 'force_st')
        self.add_fxn('dist_ee', DistEE, 'ee_1', 'ee_mot', 'ee_ctl', 'force_st')
        self.add_fxn('affect_dof', AffectDOF, 'ee_mot', 'ctl', 'des_traj',
                     'dofs', 'force_lin', ca={'p': {'archtype': self.p.arch}})
        self.add_fxn('ctl_dof', CtlDOF, 'ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path', PlanPath, 'ee_ctl', 'des_traj', 'force_st', 'dofs')
        self.add_fxn('hold_payload', HoldPayload, 'force_lin', 'force_st', 'dofs')
        self.add_fxn('view_env', ViewEnvironment, 'dofs', 'environment')


if __name__ == "__main__":
    lap = LineArchParam()
    lap = LineArchParam(archtype='quad')

    import doctest
    doctest.testmod(verbose=True)
    import multiprocessing as mp
    from fmdtools.sim.sample import FaultDomain, FaultSample
    from fmdtools.analyze.phases import PhaseMap

    # check rf_mechbreack fault propagation in quad architecture:
    hierarchical_model = Drone(p=DroneParam(arch='quad'))
    result, mdlhist = fs.propagate.one_fault(hierarchical_model,
                                             'affect_dof.ca.comps.rf',
                                             'mechbreak',
                                             time=5)
    mdlhist.plot_line('flows.dofs.s.x', 'flows.dofs.s.y', 'flows.dofs.s.z', 'fxns.store_ee.s.soc')

    # check rr2_propstuck fault in oct architecture over several times:
    mdl = Drone(p=DroneParam(arch='oct'))
    rr2_faults = FaultDomain(mdl)
    # rr2_faults.add_fault('affect_dof.ca.comps.rr2', 'propstuck')
    rr2_faults.add_all_fxn_modes("affect_dof")

    rr2_samp = FaultSample(rr2_faults, phasemap=PhaseMap(mdl.sp.phases))
    rr2_samp.add_fault_phases()

    ec, hist = fs.propagate.fault_sample(mdl, rr2_samp, staged=True)

    # plot a single scen (at t=8)
    fault_kwargs = {'alpha': 0.2, 'color': 'red'}
    h_plot = hist.get('nominal', 'drone_fxns_affect_dof_ca_comps_rr2_propstuck_t8p0').flatten()
    h_plot.plot_line('flows.dofs.s.x', 'flows.dofs.s.y', 'flows.dofs.s.z', 'fxns.store_ee.s.soc')

    # plot all scens
    hist.plot_line('flows.dofs.s.x', 'flows.dofs.s.y', 'flows.dofs.s.z', 'fxns.store_ee.s.soc',
                   indiv_kwargs={'faulty': fault_kwargs})
    fig, ax = hist.plot_trajectories("dofs.s.x", "dofs.s.y", "dofs.s.z",
                                     time_groups=['nominal'],
                                     indiv_kwargs={'faulty': fault_kwargs})

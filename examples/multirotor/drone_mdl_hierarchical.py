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
from examples.multirotor.drone_mdl_static import m2to1, DistEE, BaseLine
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, DesTraj
from examples.multirotor.drone_mdl_static import AffectDOFMode, AffectDOFState
from examples.multirotor.drone_mdl_dynamic import StoreEE, CtlDOF, PlanPath, HoldPayload
from examples.multirotor.drone_mdl_dynamic import ViewEnvironment, DroneEnvironment
from examples.multirotor.drone_mdl_dynamic import Drone as DynDrone
from examples.multirotor.drone_mdl_dynamic import AffectDOF as AffectDOFDynamic

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.block.component import Component
from fmdtools.define.architecture.component import ComponentArchitecture
from fmdtools.define.architecture.function import FunctionArchitecture

import fmdtools.sim as fs

import numpy as np

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

class LineArchState(State):
    """
    States of the line architecture.

    Fields
    ------
    forward : dict
        Correction factors for moving forward (based on which rotors are in front).
    upward: dict
        Correction factors for moving upward (1.0 unless in a recovery mode).
    """
    forward: dict = dict()
    upward: dict = dict()


class AffectDOFArch(ComponentArchitecture):
    container_p = LineArchParam
    container_s = LineArchState

    def init_architecture(self, **kwargs):
        # add lines
        for compname in self.p.components:
            self.add_comp(compname, Line)
        # add state configuration - relative throttle for each line
        if self.p.archtype == 'quad':
            self.s.forward.update({'rf': 0.5, 'lf': 0.5, 'lr': -0.5, 'rr': -0.5})
        elif self.p.archtype == 'hex':
            self.s.forward.update({'rf': 0.5, 'lf': 0.5, 'lr': -0.5,
                                   'rr': -0.5, 'r': -0.75, 'f': 0.75})
        elif self.p.archtype == 'oct':
            self.s.forward.update({'rf': 0.5, 'lf': 0.5, 'lr': -0.5, 'rr': -0.5,
                                   'rf2': 0.5, 'lf2': 0.5, 'lr2': -0.5, 'rr2': -0.5})
        self.s.upward = {c: 1.0 for c in self.p.components}


class AffectDOF(AffectDOFDynamic):
    """Rotor locomotion (multi-component extension)."""

    __slots__ = ()
    container_s = OverallAffectDOFState
    arch_ca = AffectDOFArch

    def static_behavior(self, time):
        """Rotor dynamic behavior with architecture-base recovery."""
        if self.ca.p.opposite:
            self.reconfig_faults()
        self.calc_pwr()

    def reconfig_faults(self):
        """Corrects for individual line faultmodes by turning off the opposite rotor
        and upping the throttle (amp_factor)"""
        for fault in self.m.faults:
            if fault in self.ca.m.sub_modes:
                comp = self.ca.m.sub_modes[fault]
                opp = self.ca.p.opposite[comp]
                if self.ca.s.forward[comp] != 0.0:
                    self.ca.s.forward[comp] = 0.0
                    self.ca.s.upward[comp] = 0.0
                if self.ca.s.forward[opp] != 0.0:
                    self.ca.s.forward[opp] = 0.0
                    self.ca.s.upward[opp] = 0.0
        tot_comps = len(self.ca.comps)
        empty_comps = len([c for c in self.ca.s.forward if self.ca.s.forward[c] == 0.0])
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
        for linname, lin in self.ca.comps.items():
            a, ee = lin.behavior(self.ee_in.s.effort,
                                 self.ctl_in,
                                 self.ca.s.upward[linname] * self.s.amp_factor,
                                 self.ca.s.forward[linname] * self.s.amp_factor,
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

        self.s.lrstab = (sum([air[comp] for comp in self.ca.p.lr_dict['l']]) -
                         sum([air[comp] for comp in self.ca.p.lr_dict['r']]))/len(air)
        self.s.frstab = (sum([air[comp] for comp in self.ca.p.fr_dict['r']]) -
                         sum([air[comp] for comp in self.ca.p.fr_dict['f']]))/len(air)
        if abs(self.s.lrstab) >= 0.4 or abs(self.s.frstab) >= 0.75:
            self.dofs.s.put(uppwr=0.0, planpwr=0.0)
        else:
            airs = list(air.values())
            self.dofs.s.uppwr = np.mean(airs)
            self.dofs.s.planpwr = -2*self.s.frstab


class Line(Component, BaseLine):
    """Individual version of a line (extends BaseLine in static model)."""

    __slots__ = ()
    container_s = AffectDOFState
    container_m = AffectDOFMode

    def behavior(self, ee_in, ctlin, u_fact, f_fact, force):
        """Calculate air, ee out based on inputs and modes."""
        if force <= 0.0:
            self.m.add_fault('mechbreak', 'propbreak')
        elif force <= 0.5:
            self.m.add_fault('mechfriction')

        self.calc_faults()

        pwr = ctlin.s.upward * u_fact + ctlin.s.forward * f_fact
        airout = m2to1([ee_in,
                        self.s.e_ti,
                        pwr,
                        self.s.ct,
                        self.s.mt,
                        self.s.pt])
        ee_in = m2to1([ee_in, self.s.e_to, pwr])
        return airout, ee_in


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
    endclass, mdlhist = fs.propagate.one_fault(hierarchical_model,
                                               'affect_dof',
                                               'rf_mechbreak',
                                               time=5)
    mdlhist.plot_line('flows.dofs.s.x', 'flows.dofs.s.y', 'flows.dofs.s.z', 'fxns.store_ee.s.soc')

    # check rr2_propstuck fault in oct architecture over several times:
    mdl = Drone(p=DroneParam(arch='oct'))
    rr2_faults = FaultDomain(mdl)
    rr2_faults.add_fault('affect_dof', 'rr2_propstuck')

    rr2_samp = FaultSample(rr2_faults, phasemap=PhaseMap(mdl.sp.phases))
    rr2_samp.add_fault_phases()

    ec, hist = fs.propagate.fault_sample(mdl, rr2_samp , staged=True, pool=mp.Pool(4))

    # plot a single scen (at t=8)
    fault_kwargs = {'alpha': 0.2, 'color': 'red'}
    h_plot = hist.get('nominal', 'affect_dof_rr2_propstuck_t8p0').flatten()
    h_plot.plot_line('flows.dofs.s.x', 'flows.dofs.s.y', 'flows.dofs.s.z', 'fxns.store_ee.s.soc')

    # plot all scens
    hist.plot_line('flows.dofs.s.x', 'flows.dofs.s.y', 'flows.dofs.s.z', 'fxns.store_ee.s.soc',
                   indiv_kwargs={'faulty': fault_kwargs})
    fig, ax = hist.plot_trajectories("dofs.s.x", "dofs.s.y", "dofs.s.z",
                                     time_groups=['nominal'],
                                     indiv_kwargs={'faulty': fault_kwargs})

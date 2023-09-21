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
from fmdtools.define.block import FxnBlock, Component, CompArch
from fmdtools.sim.approach import SampleApproach
from fmdtools.define.model import Model

from examples.multirotor.drone_mdl_static import m2to1, HoldPayload, DistEE, BaseLine
from examples.multirotor.drone_mdl_static import Force, EE, Control, DOFs, DesTraj
from examples.multirotor.drone_mdl_dynamic import StoreEE, CtlDOF, PlanPath, ViewEnvironment, DroneEnvironment
from examples.multirotor.drone_mdl_static import AffectDOFMode, AffectDOFState
from examples.multirotor.drone_mdl_dynamic import Drone as DynDrone
from drone_mdl_dynamic import AffectDOF as AffectDOFDynamic

class OverallAffectDOFState(State):
    lrstab: float = 0.0
    frstab: float = 0.0
    amp_factor: float = 1.0
    
    """
    Overall states for the dynamics. Has entries:
        - lrstab: float
            Left/Right stability (nominal value = 0.0)
        - frstab: float
            Front/Rear stability (nominal value = 0.0)
    """


class AffectDOFArch(CompArch):
    archtype: str = 'quad'
    forward: dict = dict()
    upward: dict = dict()
    lr_dict: dict = dict()
    fr_dict: dict = dict()
    opposite: dict = dict()
    """
    Line Architecture defined by parameter 'archtype'. Has options:
        - 'quad': quadrotor architecture
        - 'hex': hexarotor architecture
        - 'oct': octorotor architecture
    """

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
    _init_s = OverallAffectDOFState
    _init_c = AffectDOFArch

    def behavior(self, time):
        if self.c.opposite:
            self.reconfig_faults()
        self.calc_pwr()

    def reconfig_faults(self):
        """Corrects for individual line faultmodes by turning off the opposite rotor
        and upping the throttle (amp_factor)"""
        for fault in self.m.faults:
            if fault in self.c.faultmodes:
                comp = self.c.faultmodes[fault]
                opp = self.c.opposite[comp]
                if self.c.forward[comp] != 0.0:
                    self.c.forward[comp] = 0.0
                    self.c.upward[comp] = 0.0
                if self.c.forward[opp] != 0.0:
                    self.c.forward[opp] = 0.0
                    self.c.upward[opp] = 0.0
        tot_comps = len(self.c.components)
        empty_comps = len([c for c in self.c.forward if self.c.forward[c] == 0.0])
        self.s.amp_factor = tot_comps / (tot_comps - empty_comps)

    def calc_pwr(self):
        air, ee_in = {}, {}
        # injects faults into lines
        for linname, lin in self.c.components.items():
            a, ee = lin.behavior(self.ee_in.s.effort,
                                 self.ctl_in,
                                 self.c.upward[linname] * self.s.amp_factor,
                                 self.c.forward[linname] * self.s.amp_factor,
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

        self.s.lrstab = (sum([air[comp] for comp in self.c.lr_dict['l']]) -
                         sum([air[comp] for comp in self.c.lr_dict['r']]))/len(air)
        self.s.frstab = (sum([air[comp] for comp in self.c.fr_dict['r']]) -
                         sum([air[comp] for comp in self.c.fr_dict['f']]))/len(air)
        if abs(self.s.lrstab) >= 0.4 or abs(self.s.frstab) >= 0.75:
            self.dofs.s.put(uppwr=0.0, planpwr=0.0)
        else:
            airs = list(air.values())
            self.dofs.s.uppwr = np.mean(airs)
            self.dofs.s.planpwr = -2*self.s.frstab


class Line(Component, BaseLine):
    _init_s = AffectDOFState
    _init_m = AffectDOFMode

    def behavior(self, ee_in, ctlin, u_fact, f_fact, force):
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
    arch: str = 'quad'
    arch_set = ('quad', 'oct', 'hex')


class Drone(DynDrone):
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
                     'dofs', 'force_lin', c={'archtype': self.p.arch})
        self.add_fxn('ctl_dof', CtlDOF, 'ee_ctl', 'des_traj', 'ctl', 'dofs', 'force_st')
        self.add_fxn('plan_path', PlanPath, 'ee_ctl', 'des_traj', 'force_st', 'dofs')
        self.add_fxn('hold_payload', HoldPayload, 'force_lin', 'force_st', 'dofs')
        self.add_fxn('view_env', ViewEnvironment, 'dofs', 'environment')

        self.build()


if __name__ == "__main__":
    import multiprocessing as mp

    hierarchical_model = Drone(p=DroneParam(arch='quad'))
    endclass, mdlhist = fs.propagate.one_fault(hierarchical_model,
                                               'affect_dof',
                                               'rf_mechbreak',
                                               time=5)
    an.plot.hist(mdlhist, 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc')

    mdl = Drone(p=DroneParam(arch='oct'))
    app = SampleApproach(mdl, faults=[('affect_dof', 'rr2_propstuck')])
    endclasses, mdlhists = fs.propagate.approach(mdl,
                                                 app,
                                                 staged=True,
                                                 pool=mp.Pool(4))

    fault_kwargs = {'alpha': 0.2, 'color': 'red'}
    an.plot.hist(mdlhists.get('nominal', 'affect_dof_rr2_propstuck_t49p0').flatten(),
                 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc')

    an.plot.hist(mdlhists, 'flows.dofs.s.x', 'dofs.s.y', 'dofs.s.z', 'store_ee.s.soc',
                 indiv_kwargs={'faulty': fault_kwargs})
    fig, ax = an.show.trajectories(mdlhists,
                                   "dofs.s.x", "dofs.s.y", "dofs.s.z",
                                   time_groups=['nominal'],
                                   indiv_kwargs={'faulty': fault_kwargs})
    mdlhists.affect_dof_rr2_propstuck_t9p0.flows.dofs.s.z

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dynamical implementation of a tank system with contingency management.

The functions of the system are:
    - ImportLiquid (Inlet Valve)
    - StoreLiquid (Tank)
    - Export Liquid (Outlet Valve)
The Tank stores a set amount of water, the level of which is controlled by 
inlet and outlet valves. 

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

from examples.tank.tank_model import TransportLiquidState, Signal, Liquid
from examples.tank.tank_model import StoreLiquidMode 

from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.define.container.mode import Mode
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.block.function import Function

import numpy as np


class TankParam(Parameter, readonly=True):
    capacity: np.float64 = np.float64(20.0)
    turnup: np.float64 = np.float64(1.0)
    faultpolicy: tuple = tuple((a-1, b-1, c-1, ul, 0)
                               for ul in ["l", "u"]
                               for a, b, c in np.ndindex((3, 3, 3)))
    policymap: dict = dict()

    def __init__(self, *args, **kwargs):
        args = self.get_true_fields(*args, **kwargs)
        super().__init__(*args, strict_immutability=False,
                         check_type=False, check_pickle=False)
        if not self.policymap:
            self.policymap.update(self.get_faultpolicy())

    def get_faultpolicy(self):
        fd = {(v[0], v[1], v[2], v[3]): v[4] for v in self.faultpolicy}
        fp = {(a-1, b-1, c-1):
              (fd[a-1, b-1, c-1, "l"], fd[a-1, b-1, c-1, "u"])
              for a, b, c in np.ndindex((3, 3, 3))}
        return fp


def x_to_fp(*x):
    a = 1
    fp = tuple((v[0], v[1], v[2], v[3], int(x[i]))
               for i, v in enumerate(TankParam.__defaults__['faultpolicy']))
    return fp



def make_tankparam(*args, **kwargs):
    if args:
        fp = tuple((v[0], v[1], v[2], v[3], args[i])
                   for i, v in enumerate(TankParam.__defaults__['faultpolicy']))
        kwargs['faultpolicy'] = fp
    return kwargs


class TransportLiquidMode(Mode):
    fm_args = {'stuck': (1e-5,),
               'blockage': (1e-5,)}
    phases = {'na': 1.0}
    units = 'hr'


class ImportLiquid(Function):
    __slots__ = ('sig', 'wat_out')
    container_p = TankParam
    container_s = TransportLiquidState
    container_m = TransportLiquidMode
    flow_sig = Signal
    flow_wat_out = Liquid
    flownames = {'coolant_in': 'wat_out', 'input_sig': 'sig'}

    def static_behavior(self, time):
        if self.sig.s.action >= 1:
            self.s.amt_open = 1 + self.p.turnup
        elif self.sig.s.action == 0:
            self.s.amt_open = 1
        elif self.sig.s.action == -1:
            self.s.amt_open = 0

        if self.m.has_fault('blockage'):
            self.wat_out.s.effort = 0.0
            self.sig.s.indicator = 1
        elif self.m.has_fault('leak'):
            self.wat_out.s.effort = self.s.amt_open - 1.0
            self.sig.s.indicator = -1
        else:
            self.wat_out.s.effort = self.s.amt_open
            self.sig.s.indicator = 0


class ExportLiquid(Function):
    __slots__ = ('sig', 'wat_in')
    container_p = TankParam
    container_s = TransportLiquidState
    container_m = TransportLiquidMode
    flow_sig = Signal
    flow_wat_in = Liquid
    flownames = {'coolant_out': 'wat_in', 'output_sig': 'sig'}

    def static_behavior(self, time):
        if self.sig.s.action >= 1:
            self.s.amt_open = 1 + self.p.turnup
        elif self.sig.s.action == 0:
            self.s.amt_open = 1
        elif self.sig.s.action == -1:
            self.s.amt_open = 0

        if self.m.has_fault('blockage'):
            self.wat_in.s.rate = 0.0
            self.sig.s.indicator = 1
        elif self.m.has_fault('leak'):
            self.wat_in.s.rate = self.s.amt_open*self.wat_in.s.effort + 1.0
            self.sig.s.indicator = -1
        else:
            self.wat_in.s.rate = self.s.amt_open*self.wat_in.s.effort
            self.sig.s.indicator = 0


class StoreLiquidState(State):
    level: float = 10.0
    net_flow: float = 0.0
    coolingbuffer: float = 10.0


class StoreLiquid(Function):
    __slots__ = ('wat_in', 'wat_out', 'sig')
    container_s = StoreLiquidState
    container_m = StoreLiquidMode
    container_p = TankParam
    flow_wat_in = Liquid
    flow_wat_out = Liquid
    flow_sig = Signal
    flownames = {'coolant_in': 'wat_in', 'coolant_out': 'wat_out', 'tank_sig': 'sig'}

    def static_behavior(self, time):
        if self.s.level >= self.p.capacity:
            self.wat_in.s.rate = 0.0 * self.wat_in.s.effort
            self.wat_out.s.effort = 2.0 * self.wat_in.s.effort
            self.s.level = self.p.capacity
        elif self.s.level <= 0.0:
            self.wat_out.s.effort = 0.0
            self.wat_in.s.rate = self.wat_in.s.effort
        else:
            self.wat_in.s.rate = self.wat_in.s.effort
            self.wat_out.s.effort = 1.0
        if self.s.level > self.p.capacity/2+5:
            self.sig.s.indicator = -1
        elif self.s.level < self.p.capacity/2-5:
            self.sig.s.indicator = 1
        else:
            self.sig.s.indicator = 0

        if self.m.has_fault('leak'):
            self.s.net_flow = self.wat_in.s.rate - self.wat_out.s.rate - 1.0
        else:
            self.s.net_flow = self.wat_in.s.rate - self.wat_out.s.rate

    def dynamic_behavior(self, time):
        self.s.inc(level=self.s.net_flow)
        self.s.coolingbuffer = max(self.s.coolingbuffer - 1.0 + self.wat_in.s.rate, 0)


class ContingencyActions(Function):

    __slots__ = ('input_sig', 'output_sig', 'tank_sig')
    container_p = TankParam
    flow_input_sig = Signal
    flow_output_sig = Signal
    flow_tank_sig = Signal

    def dynamic_behavior(self, time):
        self.input_sig.s.action = self.p.policymap[self.input_sig.s.indicator,
                                                   self.tank_sig.s.indicator,
                                                   self.output_sig.s.indicator][0]
        self.output_sig.s.action = self.p.policymap[self.input_sig.s.indicator,
                                                    self.tank_sig.s.indicator,
                                                    self.output_sig.s.indicator][1]


class Tank(FunctionArchitecture):
    __slots__ = ()
    container_p = TankParam
    default_sp = dict(phases=(('na', 0, 0),
                              ('operation', 1, 20)),
                      end_time=20,
                      units='min')

    def init_architecture(self, **kwargs):
        self.add_flow('coolant_in', Liquid)
        self.add_flow('coolant_out', Liquid)
        self.add_flow('input_sig', Signal)
        self.add_flow('tank_sig', Signal)
        self.add_flow('output_sig', Signal)

        self.add_fxn('import_coolant', ImportLiquid,
                     'coolant_in', 'input_sig', p=self.p)
        self.add_fxn('store_coolant', StoreLiquid,
                     'coolant_in', 'coolant_out', 'tank_sig',
                     p=self.p,
                     s={'level': self.p.capacity/2, 'coolingbuffer': self.p.capacity/2})

        self.add_fxn('export_coolant', ExportLiquid,
                     'coolant_out', 'output_sig',
                     p=self.p)
        self.add_fxn('contingency', ContingencyActions,
                     'input_sig', 'tank_sig', 'output_sig', p=self.p)

    def find_classification(self, scen, mdlhists):
        # here we define failure in terms of the water level getting too low or too high
        overfullcost, emptycost, buffercost = 0, 0, 0
        # calculate time the tank is overfull:
        sum(self.h.fxns.store_coolant.s.level >= self.p.capacity)*10000
        # cost if the tank lacks any water:
        if any(self.h.fxns.store_coolant.s.level <= 0):
            emptycost = 1000000
        # cost if the buffer is 'spent'
        buffercost = sum(self.h.fxns.store_coolant.s.coolingbuffer <= 0)*100000
        mitigationcost = (sum(self.h.flows.input_sig.s.action != 0) +
                          sum(self.h.flows.output_sig.s.action != 0))*1000
        totcost = overfullcost + emptycost + buffercost + mitigationcost
        rate = scen.rate
        life = 1e5
        return {'rate': rate, 'cost': totcost, 'expected_cost': rate*life*totcost}


if __name__ == "__main__":
    import fmdtools.sim.propagate as propagate

    mdl = Tank()
    vals = ['fxns.store_coolant.s.level',
            'fxns.store_coolant.s.net_flow',
            'fxns.store_coolant.s.coolingbuffer']
    endresults, mdlhist = propagate.nominal(mdl, desired_result=['endclass', 'graph'])
    mdlhist.plot_line(*vals)

    # check faulty run
    result, mdlhist = propagate.one_fault(mdl, 'export_coolant', 'blockage', time=2,
                                          desired_result=['endclass', 'graph'])

    mdlhist.plot_line(*vals, time_slice=2, title='NotVisible')
    result.graph.draw(title='NotVisible, time=2')
    result.graph.draw_graphviz(title='NotVisible, time=2')

    from fmdtools.sim.sample import ParameterDomain
    pd = ParameterDomain(TankParam)
    pd.add_variables("capacity", "turnup")
    fp_vars = [1 for i, v in enumerate(TankParam.__defaults__['faultpolicy'])]
    fp_varnames = ['faultpolicy.'+str(i) for i, v in enumerate(TankParam.__defaults__['faultpolicy'])]
    x_to_fp(*fp_vars)
    pd = ParameterDomain(TankParam)
    pd.add_variables("capacity", "turnup")
    pd.add_variables(*fp_varnames, var_map=x_to_fp)
    pd(1, 1, *fp_vars)

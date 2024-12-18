#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple model for explaining stochastic behavior modelling

This model is an extension of ex_pump.py that includes stochastic behaviors

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

from examples.pump.ex_pump import MoveWat as DetMoveWat
from examples.pump.ex_pump import ImportWater, ExportWater
from examples.pump.ex_pump import ImportEE as DetImportEE
from examples.pump.ex_pump import ImportSig as DetImportSig
from examples.pump.ex_pump import PumpParam, Electricity, Water, Signal
from examples.pump.ex_pump import Pump as DetPump

from fmdtools.define.container.rand import Rand
from fmdtools.define.container.state import State
import fmdtools.sim.propagate as propagate
import numpy as np


class ImportEERandState(State):
    effstate: float = 1.0
    grid_noise: float = 1.0


class ImportEERand(Rand, copy_default=True):
    s: ImportEERandState = ImportEERandState()


class ImportEE(DetImportEE):

    __slots__ = ()
    container_r = ImportEERand

    def set_faults(self):
        if self.ee_out.s.current > 20.0:
            self.m.add_fault('no_v')

    def static_behavior(self, time):
        self.set_faults()
        if self.m.has_fault('no_v'):
            self.r.s.effstate = 0.0  # an open circuit means no voltage is exported
        elif self.m.has_fault('inf_v'):
            self.r.s.effstate = 100.0  # a voltage spike means voltage is much higher
        else:
            if time > self.t.time:
                self.r.set_rand_state('effstate', 'triangular', 0.9, 1, 1.1)
        if time > self.t.time:
            self.r.set_rand_state('grid_noise', 'normal',
                                  1, 0.05*(2+np.sin(np.pi/2*time)))
        self.ee_out.s.voltage = self.r.s.grid_noise*self.r.s.effstate * 500


class ImportSigRandState(State):
    sig_noise: float = 1.0


class ImportSigRand(Rand):
    s: ImportSigRandState = ImportSigRandState()


class ImportSig(DetImportSig):

    __slots__ = ()
    container_r = ImportSigRand

    def static_behavior(self, time):
        if self.m.has_fault('no_sig'):
            self.sig_out.s.power = 0.0  # an open circuit means no voltage is exported
        else:
            if time < 5:
                self.sig_out.s.power = 0.0
                self.r.s.to_default('sig_noise')
            elif time < 50:
                if not time % 5:
                    self.r.set_rand_state('sig_noise', 'choice', [1.0, 0.9, 1.1])
                self.sig_out.s.power = 1.0*self.r.s.sig_noise
            else:
                self.sig_out.s.power = 0.0
                self.r.s.to_default()


class MoveWatStates(State):
    total_flow: float = 0.0
    eff: float = 1.0  # effectiveness state


class MoveWatRandState(State):
    eff: float = 1.0
    eff_update = ('normal', (1.0, 0.2))


class MoveWatRand(Rand):
    s: MoveWatRandState = MoveWatRandState()


class MoveWat(DetMoveWat):

    __slots__ = ()
    container_s = MoveWatStates
    container_r = MoveWatRand

    def static_behavior(self, time):
        self.s.eff = self.r.s.eff
        super().static_behavior(time)
        if time > self.t.time:
            self.s.inc(total_flow=self.wat_out.s.flowrate)


class Pump(DetPump):

    __slots__ = ()
    default_track = 'all'
    container_r = Rand

    def init_architecture(self, **kwargs):
        self.add_flow('ee_1', Electricity)
        self.add_flow('sig_1', Signal)
        self.add_flow('wat_1', Water('Wat_1'))
        self.add_flow('wat_2', Water('Wat_1'))

        self.add_fxn('import_ee', ImportEE, 'ee_1')
        self.add_fxn('import_water', ImportWater, 'wat_1')
        self.add_fxn('import_signal', ImportSig, 'sig_1')
        self.add_fxn('move_water', MoveWat, 'ee_1', 'sig_1', 'wat_1', 'wat_2',
                     p={'delay': self.p.delay})
        self.add_fxn('export_water', ExportWater, 'wat_2')


if __name__ == "__main__":
    import multiprocessing as mp
    from fmdtools.sim.sample import ParameterDomain, ParameterSample

    # convert to tests 1:
    mdl = Pump()

    # now tested in test_plot_nested_hist
    pd = ParameterDomain(PumpParam)
    pd.add_variable("delay")

    ps = ParameterSample(pd)
    ps.add_variable_replicates([[1]], replicates=10, name="delay1")
    ps.add_variable_replicates([[10]], replicates=10, name="delay10")

    faultdomains = {'fd': (('fault', 'export_water', 'block'), {})}
    faultsamples = {'fs': (('fault_phases', 'fd'), {})}

    pool = mp.Pool(4)
    ecs, hists, apps = propagate.nested_sample(mdl, ps, run_stochastic=True,
                                               faultdomains=faultdomains,
                                               faultsamples=faultsamples,
                                               pool=pool)

    # convert to plot tests:
    comp_mdlhists = hists.get_scens('export_water_block_t27p0')
    comp_groups = {'delay_1': ps.get_scens(p={'delay': 2}),
                   'delay_10': ps.get_scens(p={'delay': 10})}
    comp_mdlhists.plot_line('fxns.move_water.s.eff',
                            'fxns.move_water.s.total_flow',
                            'flows.wat_2.s.flowrate',
                            'flows.wat_2.s.pressure',
                            comp_groups=comp_groups,
                            aggregation='percentile',
                            time_slice=27)

    comp_mdlhists.plot_line('fxns.move_water.s.eff',
                           'fxns.move_water.s.eff',
                           'fxns.move_water.s.total_flow',
                           'flows.wat_2.s.flowrate',
                           'flows.wat_2.s.pressure',
                           aggregation='percentile', time_slice=27)

    comp_mdlhists.plot_metric_dist([5,10,15],
                                   'fxns.move_water.s.eff',
                                   'fxns.move_water.s.total_flow',
                                   'flows.wat_2.s.flowrate',
                                   'flows.wat_2.s.pressure')

    ani = comp_mdlhists.animate('plot_metric_dist_from',
                                plot_values=('fxns.move_water.s.eff',
                                             'fxns.move_water.s.total_flow',
                                             'flows.wat_2.s.flowrate',
                                             'flows.wat_2.s.pressure'))

    ps2 = ParameterSample(pd)
    ps2.add_variable_replicates([[0]], replicates=100, name="nodelay")
    ps2.add_variable_replicates([[10]], replicates=100, name="delay10")
    nomhist, nomres, = propagate.parameter_sample(mdl, ps)
    ps2.group_scens("p.delay")

    # test prob dense?
    for i in range(1, 10):
        mdl.update_seed(i)
        mdl.propagate(i, run_stochastic='track_pdf')
        print(mdl.return_probdens())
        #print(mdl.seed)

        #for fxnname, fxn in mdl.fxns.items():
        #    print(fxnname+': ')
        #    print(fxn.return_probdens())
        #    print(getattr(fxn, 'pds', None))

    endresults, mdlhist = propagate.one_fault(mdl, 'export_water', 'block',
                                              time=20, staged=False, run_stochastic=False,
                                              new_params={'modelparams': {'seed': 50}})

    endresults, mdlhist = propagate.one_fault(mdl, 'export_water', 'block',
                                              time=20, staged=False, run_stochastic=True,
                                              new_params={'modelparams': {'seed': 50}})

    # mdlhist['faulty']['functions']['ImportEE']['probdens']

    mdlhist.plot_line('fxns.import_ee.s.effstate', 'fxns.import_ee.r.s.grid_noise',
                      'flows.ee_1.s.voltage', 'flows.ee_1.s.current')

    # an.plot.mdlhists(mdlhist, fxnflowvals={'ImportEE'})


# -*- coding: utf-8 -*-
"""
File name: pump_stochastic.py
Author: Daniel Hulse
Created: October 2019
Description: A simple model for explaining stochastic behavior modelling

This model is an extension of ex_pump.py that includes stochastic behaviors
"""
from examples.pump.ex_pump  import MoveWat as DetMoveWat
from fmdtools.define.rand import Rand
from fmdtools.define.state import State
import fmdtools.analyze as an
import fmdtools.sim.propagate as propagate
import numpy as np


from examples.pump.ex_pump import ImportWater, ExportWater
from examples.pump.ex_pump import ImportEE as DetImportEE


class ImportEERandState(State):
    effstate: float = 1.0
    grid_noise: float = 1.0


class ImportEERand(Rand):
    s = ImportEERandState()


class ImportEE(DetImportEE):
    __slots__ = ()
    _init_r = ImportEERand

    def condfaults(self, time):
        if self.ee_out.s.current > 20.0:
            self.m.add_fault('no_v')

    def behavior(self, time):
        if self.m.has_fault('no_v'):
            self.r.s.effstate = 0.0  # an open circuit means no voltage is exported
        elif self.m.has_fault('inf_v'):
            self.r.s.effstate = 100.0  # a voltage spike means voltage is much higher
        else:
            if time > self.t.time:
                self.r.set_rand('effstate', 'triangular', 0.9, 1, 1.1)
        if time > self.t.time:
            self.r.set_rand('grid_noise', 'normal', 1, 0.05*(2+np.sin(np.pi/2*time)))
        self.ee_out.s.voltage = self.r.s.grid_noise*self.r.s.effstate * 500


from examples.pump.ex_pump import ImportSig as DetImportSig


class ImportSigRandState(State):
    sig_noise: float = 1.0


class ImportSigRand(Rand):
    s = ImportSigRandState()


class ImportSig(DetImportSig):
    __slots__ = ()
    _init_r = ImportSigRand

    def behavior(self, time):
        if self.m.has_fault('no_sig'):
            self.sig_out.power = 0.0  # an open circuit means no voltage is exported
        else:
            if time < 5:
                self.sig_out.power = 0.0
                self.r.to_default('sig_noise')
            elif time < 50:
                if not time % 5:
                    self.r.set_rand('sig_noise', 'choice', [1.0, 0.9, 1.1])
                self.sig_out.power = 1.0*self.r.s.sig_noise
            else:
                self.sig_out.power = 0.0
                self.r.to_default()


class MoveWatStates(State):
    total_flow: float = 0.0
    eff: float = 1.0  # effectiveness state


class MoveWatRandState(State):
    eff: float = 1.0
    eff_update = ('normal', (1.0, 0.2))


class MoveWatRand(Rand):
    s = MoveWatRandState()


class MoveWat(DetMoveWat):
    __slots__ = ()
    _init_s = MoveWatStates
    _init_r = MoveWatRand

    def behavior(self, time):
        self.s.eff = self.r.s.eff
        super().behavior(time)
        if time > self.t.time:
            self.s.inc(total_flow=self.wat_out.s.flowrate)


from examples.pump.ex_pump import PumpParam, Electricity, Water, Signal
from examples.pump.ex_pump import Pump as DetPump


class Pump(DetPump):
    __slots__ = ()
    default_track = 'all'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        self.build()


if __name__ == "__main__":
    import multiprocessing as mp
    from fmdtools.define.model import check_model_pickleability
    from fmdtools.define.common import check_pickleability
    from fmdtools.sim.sample import ParameterDomain, ParameterSample

    # convert to tests 1:
    mdl = Pump()
    """
    app = NominalApproach()
    app.add_seed_replicates('test_seeds', 100)
    
    endclasses, mdlhists=propagate.nominal_approach(mdl,app, run_stochastic=True)
    
    fig = mdlhists.plot_line('move_water.r.s.eff', 'move_water.s.total_flow',
                             'wat_2.s.flowrate', 'wat_2.s.pressure',
                             'import_ee.r.s.effstate', 'import_ee.r.s.grid_noise',
                             'ee_1.s.voltage', 'sig_1.s.power',
                             color='blue', comp_groups={}, aggregation='percentile')
    
    fig = mdlhists.plot_line( 'move_water.r.s.eff', 'move_water.s.total_flow',
                             'wat_2.s.flowrate', 'wat_2.s.pressure',
                             'import_ee.r.s.effstate', 'import_ee.r.s.grid_noise',
                             'ee_1.s.voltage', 'sig_1.s.power',
                             color='blue', comp_groups={}, aggregation='mean_ci')
    
    # convert to test 2:
    rp = SimParam(phases=(('start',0,4),('on',5,49),('end',50,55)), times=(0,20, 55), dt=1.0, units='hr')
    mdl = Pump(sp = rp, r={'seed':5})
    
    check_model_pickleability(mdl, try_pick=True)
    
    mdl.set_vars([['ee_1', 'current']],[2])
    """
    # convert to test 3:
    pd = ParameterDomain(PumpParam)
    pd.add_variable("delay")

    ps = ParameterSample(pd)
    ps.add_variable_replicates([[1]], replicates=10, name="delay1")
    ps.add_variable_replicates([[10]], replicates=10, name="delay10")

    faultdomains = {'fd': (('fault', 'export_water', 'block'), {})}
    faultsamples = {'fs': (('fault_phases', 'fd'), {})}

    ecs, hists, apps = propagate.nested_sample(mdl, ps, run_stochastic=True,
                                               faultdomains=faultdomains,
                                               faultsamples=faultsamples,
                                               pool=mp.Pool(4))

    from fmdtools.analyze.tabulate import NominalEnvelope
    ne = NominalEnvelope(ps, ecs, 'cost', 'p.delay')

    ne.as_plot()
    # convert to plot tests:
    comp_mdlhists = hists.get_scens('export_water_block_t27p0')
    comp_groups = {'delay_1': ps.get_scens(p={'delay': 1}),
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

    ps2 = ParameterSample(pd)
    ps2.add_variable_replicates([[0]], replicates=100, name="nodelay")
    ps2.add_variable_replicates([[10]], replicates=100, name="delay10")
    nomhist, nomres, = propagate.parameter_sample(mdl, ps)
    ps2.group_scens("inputparams")




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

    endresults,  mdlhist = propagate.one_fault(mdl, 'export_water', 'block',
                                               time=20, staged=False, run_stochastic=False,
                                               new_params={'modelparams': {'seed': 50}})

    endresults,  mdlhist = propagate.one_fault(mdl, 'export_water', 'block',
                                               time=20, staged=False, run_stochastic=True,
                                               new_params={'modelparams': {'seed': 50}})

    #mdlhist['faulty']['functions']['ImportEE']['probdens']

    mdlhist.plot_line('fxns.import_ee.s.effstate', 'fxns.import_ee.r.s.grid_noise',
                      'flows.ee_1.s.voltage', 'flows.ee_1.s.current')
    #an.plot.mdlhists(mdlhist, fxnflowvals={'ImportEE'})
    


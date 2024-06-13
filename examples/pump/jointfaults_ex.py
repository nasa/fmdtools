# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:45:41 2019

This code tests some approaches to sampling joing fault scenarios

@author: Daniel Hulse
"""
import fmdtools.sim.propagate as prop
import fmdtools.analyze as an
from fmdtools.sim.sample import FaultSample, FaultDomain
from fmdtools.analyze.phases import PhaseMap
from fmdtools.analyze.history import History
from examples.pump.ex_pump import Pump  # required to import entire module


if __name__ == "__main__":
    mdl = Pump()

    all_fd = FaultDomain(mdl)
    all_fd.add_all()

    fs_2 = FaultSample(all_fd, phasemap=PhaseMap(mdl.sp.phases))
    fs_2.add_fault_phases(n_joint=2, baserate='max', p_cond=0.1)

    # if a function can have multiple modes injected at the same time
    fs_3 = FaultSample(all_fd, phasemap=PhaseMap(mdl.sp.phases))
    fs_3.add_fault_phases(n_joint=3, baserate='max', p_cond=0.1)

    # note that faults above level 4 have very low rates, even with p_cond
    fs_5 = FaultSample(all_fd, phasemap=PhaseMap(mdl.sp.phases))
    fs_5.add_fault_phases(n_joint=5, baserate='max', p_cond=0.1)

    endclasses, mdlhists = prop.fault_sample(mdl, fs_2)

    fmea = an.tabulate.FMEA(endclasses, fs_2, group_by=('phase', 'functions', 'modes'))
    fmea.sort_by_metric("expected_cost")
    fmea.as_plot("expected_cost", color_factor='phase', suppress_ticklabels=True)

    endclasses, mdlhists = prop.fault_sample(mdl, fs_5)

    mdlhist = History({'nominal': mdlhists.get('nominal'),
               'faulty': mdlhists.get('import_ee_no_v__import_water_no_wat__import_signal_no_sig__move_water_mech_break__export_water_block_t27p0')})
    mdlhist.plot_line('flows.ee_1.s.current', 'flows.wat_2.s.flowrate', time_slice=27)

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:58:59 2023

@author: dhulse
"""

from drone_mdl_rural import Drone
from examples.eps.eps import EPS

from fmdtools.sim.approach import gen_interval_times
from recordclass import dataobject, asdict
from fmdtools.define.common import set_var, get_var, t_key
from fmdtools.sim.scenario import SingleFaultScenario

import numpy as np

mdl = Drone()
#app = SampleApproach(mdl)

def same_mode(modename1, modename2, exact=True):
    if exact:
        return modename1 == modename2
    else:
        return modename1 in modename2



class FaultDomain(object):
    """
    Defines the faults which will be sampled from in an approach.

    Attributes
    ----------
    fxns : dict
        Dict of fxns in the given Simulable (to simulate)
    faults : dict
        Dict of faults to inject in the simulable
    """

    def __init__(self, mdl):
        self.mdl = mdl
        self.fxns = mdl.get_fxns()
        self.faults = {}

    def __repr__(self):
        faultlist = [str(fault) for fault in self.faults]
        if len(faultlist) > 10:
            faultlist = faultlist[0:10] + [["...more"]]
        modestr = "FaultDomain with faults:" + "\n -" + "\n -".join(faultlist)
        return modestr

    def add_fault(self, fxnname, faultmode):
        """
        Add a fault to the FaultDomain.

        Parameters
        ----------
        fxnname : str
            Name of the simulable to inject in
        faultmode : str
            Name of the faultmode to inject.
        """
        fault = self.fxns[fxnname].m.faultmodes[faultmode]
        self.faults[((fxnname, faultmode),)] = fault

    def add_faults(self, *faults):
        """
        Add multiple faults to the FaultDomain.

        Parameters
        ----------
        *faults : tuple
            Faults (simname, faultmode) to inject

        Examples
        --------
        >>> fd= FaultDomain(Drone())
        >>> fd.add_faults(('ctl_dof', 'noctl'), ('affect_dof', 'rr_ctldn'))
        >>> fd
        FaultDomain with faults:
         -(('ctl_dof', 'noctl'),)
         -(('affect_dof', 'rr_ctldn'),)
        """
        for fault in faults:
            self.add_fault(fault[0], fault[1])

    def add_all(self):
        """
        Add all faults in the Simulable to the FaultDomain.

        Examples
        --------
        >>> fd = FaultDomain(Drone().fxns['ctl_dof'])
        >>> fd.add_all()
        >>> fd
        FaultDomain with faults:
         -(('ctl_dof', 'noctl'),)
         -(('ctl_dof', 'degctl'),)
        """
        faults = [(fxnname, mode) for fxnname, fxn in self.fxns.items()
                  for mode in fxn.m.faultmodes]
        self.add_faults(*faults)

    def add_all_modes(self, *modenames, exact=True):
        """
        Add all modes with the given modenames to the FaultDomain.

        Parameters
        ----------
        *modenames : str
            Names of the modes
        exact : bool, optional
            Whether the mode name must be an exact match. The default is True.
        """
        for modename in modenames:
            faults = [(fxnname, mode) for fxnname, fxn in self.fxns.items()
                      for mode in fxn.m.faultmodes
                      if same_mode(modename, mode, exact=exact)]
            self.add_faults(*faults)

    def add_all_fxnclass_modes(self, *fxnclasses):
        """
        Add all modes corresponding to the given fxnclasses.

        Parameters
        ----------
        *fxnclasses : str
            Name of the fxnclass (e.g., "AffectDOF", "MoveWater")

        Examples
        --------
        >>> fd1 = FaultDomain(EPS())
        >>> fd1.add_all_fxnclass_modes("ExportHE")
        >>> fd1
        FaultDomain with faults:
         -(('export_he', 'hot_sink'),)
         -(('export_he', 'ineffective_sink'),)
         -(('export_waste_h1', 'hot_sink'),)
         -(('export_waste_h1', 'ineffective_sink'),)
         -(('export_waste_ho', 'hot_sink'),)
         -(('export_waste_ho', 'ineffective_sink'),)
         -(('export_waste_hm', 'hot_sink'),)
         -(('export_waste_hm', 'ineffective_sink'),)
        """
        for fxnclass in fxnclasses:
            faults = [(fxnname, mode)
                      for fxnname, fxn in mdl.fxns_of_class(fxnclass).items()
                      for mode in fxn.m.faultmodes]
            self.add_faults(*faults)

    def add_all_fxn_modes(self, *fxnnames):
        """
        Add all modes in the given simname.

        Parameters
        ----------
        *fxnnames : str
            Names of the functions (e.g., "affect_dof", "move_water").

        Examples
        --------
        >>> fd = FaultDomain(Drone())
        >>> fd.add_all_fxn_modes("hold_payload")
        >>> fd
        FaultDomain with faults:
         -(('hold_payload', 'break'),)
         -(('hold_payload', 'deform'),)
        """
        for fxnname in fxnnames:
            faults = [(fxnname, mode) for mode in self.fxns[fxnname].m.faultmodes]
            self.add_faults(*faults)

    def add_singlecomp_modes(self, *fxns):
        """
        Add all single-component modes in functions.

        Parameters
        ----------
        *fxns : str
            Names of the functions containing the components.

        Examples
        --------
        >>> fd = FaultDomain(Drone())
        >>> fd.add_singlecomp_modes("affect_dof")
        >>> fd
        FaultDomain with faults:
         -(('affect_dof', 'lf_short'),)
         -(('affect_dof', 'lf_openc'),)
         -(('affect_dof', 'lf_ctlup'),)
         -(('affect_dof', 'lf_ctldn'),)
         -(('affect_dof', 'lf_ctlbreak'),)
         -(('affect_dof', 'lf_mechbreak'),)
         -(('affect_dof', 'lf_mechfriction'),)
         -(('affect_dof', 'lf_propwarp'),)
         -(('affect_dof', 'lf_propstuck'),)
         -(('affect_dof', 'lf_propbreak'),)
        """
        if not fxns:
            fxns = tuple(self.fxns)
        for fxn in fxns:
            if hasattr(self.fxns[fxn], 'ca'):
                firstcomp = list(self.fxns[fxn].ca.components)[0]
                compfaults = [(fxn, fmode)
                              for fmode, comp in self.fxns[fxn].ca.faultmodes.items()
                              if firstcomp == comp]
                self.add_faults(*compfaults)

    def calc_rates(self, phases={}, modephases={}):
        rates = {}
        for fault in self.faults:
            rates[fault] = self.mdl.get_scen_rate()
        return rates
    

def create_scenname(faulttup, time):
    return ' '.join([fm[0]+'_'+fm[1]+'_' for fm in faulttup])+t_key(time)

class FaultSample():
    def __init__(self, faultdomain, phasemap={}):
        self.faultdomain = faultdomain
        self.phasemap = phasemap
        self.scenlist = []
        self.times = set()

    def add_single_fault_scenario(self, faulttup, time, weight=1.0):
        self.times.add(time)
        if len(faulttup) == 1:
            faulttup = faulttup[0]
        rate = self.faultdomain.mdl.get_scen_rate(faulttup[0], faulttup[1], time,
                                                  phasemap=self.phasemap,
                                                  weight=weight)
        scen = SingleFaultScenario(function=faulttup[0],
                                   fault=faulttup[1],
                                   rate=rate,
                                   name=create_scenname((faulttup,), time),
                                   time=time)
        self.scenlist.append(scen)

    def add_single_fault_times(self, times, weights=[]):

        for faulttup in self.faultdomain.faults:
            for i, time in enumerate(times):
                if weights:
                    weight = weights[i]
                elif self.phasemap:
                    phase_samples = self.phasemap.calc_samples_in_phases(*times)
                    phase = self.phasemap.find_base_phase(time)
                    weight = 1/phase_samples[phase]
                else:
                    weight = 1.0
                self.add_single_fault_scenario(faulttup, time, weight=weight)

    def add_single_fault_phases(self, *phases_to_sample, method='even', args=(1,),
                                phase_methods={}, phase_args={}):
        if self.phasemap:
            phasetimes = self.phasemap.gen_sample_times()
        else:
            interval = [0, self.faultdomain.mdl.sp.times[-1]]
            tstep = self.faultdomain.mdl.sp.dt
            phasetimes = {'phase': gen_interval_times(interval, tstep)}

        for phase, times in phasetimes.items():
            loc_method = phase_methods.get(phase, method)
            loc_args = phase_args.get(phase, args)
            if loc_method == 'even':
                sampletimes, weights = sample_times_even(times, *loc_args)
            elif loc_method == 'quad':
                sampletimes, weights = sample_times_quad(times, *loc_args)
            else:
                raise Exception("Invalid method: "+loc_method)
            self.add_single_fault_times(sampletimes, weights)


def sample_times_even(times, numpts):
    """
    Get sample time for the number of points from sampling evenly.

    Parameters
    ----------
    times : list
        Times to sample.
    numpts : int
        Number of points to sample.

    Returns
    -------
    sampletimes : list
        List of times to sample
    weights : list
        Weights.

    Examples
    --------
    >>> sample_times_even([0,1,2,3,4], 2)
    ([1, 3], [0.5, 0.5])
    """
    if numpts+2 > len(times):
        sampletimes = times
    else:
        pts = [int(round(np.quantile(times, p/(numpts+1))))
               for p in range(numpts+2)][1:-1]
        sampletimes = [times[pt] for pt in pts]
    weights = [1/len(sampletimes) for i in sampletimes]
    return sampletimes, weights


def sample_times_quad(times, nodes, weights):
    """
    Get the sample times for the given quadrature defined by nodes and weights.

    Parameters
    ----------
    times : list
        Times to sample.
    nodes : nodes
        quadrature nodes (ranging between -1 and 1)
    weights : weights
        corresponding quadrature weights

    Returns
    -------
    sampletimes : list
        List of times to sample
    weights : list
        Weights.

    Examples
    --------
    >>> sample_times_quad([0,1,2,3,4], [-0.5, 0.5], [0.5, 0.5])
    ([1, 3], [0.5, 0.5])
    """
    quantiles = np.array(nodes)/2 + 0.5
    if len(quantiles) > len(times):
        raise Exception("Nodes length " + str(len(nodes))
                        + "longer than times" + str(len(times)))
    else:
        sampletimes = [int(round(np.quantile(times, q))) for q in quantiles]
        weights = np.array(weights)/sum(weights)
    return sampletimes, list(weights)
   

        
# 'all', 'quad': {'nodes':[], 'weights': []}, n_pts = int

#class JointFaultSample

#class 



mdl = Drone()
fd = FaultDomain(mdl)
fd.add_fault("affect_dof", "rf_propwarp")
fd.add_faults(("affect_dof", "rf_propwarp"), ("affect_dof", "lf_propwarp"))
fd.add_all_modes("propwarp")

fs = FaultSample(fd)
fs.add_single_fault_scenario(("affect_dof", "rf_propwarp"), 5)
fs.add_single_fault_times([1,2,3])

from examples.eps.eps import EPS
mdl = EPS()
fd1 = FaultDomain(mdl)
fd1.add_all_fxnclass_modes("ExportHE")

# faults
# phases, modephases -> rates/probs
# phase sampling type
# joint faults
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

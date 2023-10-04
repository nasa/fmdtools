# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:58:59 2023

@author: dhulse
"""

from drone_mdl_rural import Drone
from examples.eps.eps import EPS

from fmdtools.sim.approach import SampleApproach
from recordclass import dataobject, asdict
from fmdtools.define.common import set_var, get_var

mdl = Drone()
#app = SampleApproach(mdl)

# class ParamDomain

# class StateDomain

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
        self.fxns = mdl.get_fxns()
        self.faults = {}

    def __repr__(self):
        faultlist = list(self.faults)
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
        self.faults[fxnname+".m.faults."+faultmode] = fault

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
         -ctl_dof.m.faults.noctl
         -affect_dof.m.faults.rr_ctldn
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
         -ctl_dof.m.faults.noctl
         -ctl_dof.m.faults.degctl
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
         -export_he.m.faults.hot_sink
         -export_he.m.faults.ineffective_sink
         -export_waste_h1.m.faults.hot_sink
         -export_waste_h1.m.faults.ineffective_sink
         -export_waste_ho.m.faults.hot_sink
         -export_waste_ho.m.faults.ineffective_sink
         -export_waste_hm.m.faults.hot_sink
         -export_waste_hm.m.faults.ineffective_sink
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
         -hold_payload.m.faults.break
         -hold_payload.m.faults.deform
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
         -affect_dof.m.faults.lf_short
         -affect_dof.m.faults.lf_openc
         -affect_dof.m.faults.lf_ctlup
         -affect_dof.m.faults.lf_ctldn
         -affect_dof.m.faults.lf_ctlbreak
         -affect_dof.m.faults.lf_mechbreak
         -affect_dof.m.faults.lf_mechfriction
         -affect_dof.m.faults.lf_propwarp
         -affect_dof.m.faults.lf_propstuck
         -affect_dof.m.faults.lf_propbreak
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




mdl = Drone()
fd = FaultDomain(mdl)
fd.add_fault("affect_dof", "rf_propwarp")
fd.add_faults(("affect_dof", "rf_propwarp"), ("affect_dof", "lf_propwarp"))
fd.add_all_modes("propwarp")


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
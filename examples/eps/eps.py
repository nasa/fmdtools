# -*- coding: utf-8 -*-
"""
EPS Model 
This electrical power system model showcases how fmdtools can be used for purely static 
propogation models (where the dynamic states are not a concern). This EPS system was
previously provided in the IBFM fault modelling toolkit (see: https://github.com/DesignEngrLab/IBFM ) 
and other references--this implementation follows the simple_eps model in IBFM.
    
The main purpose of this system is to supply power to optical, mechanical, and heat loads.
In this model, we represent the failure behavior of the system at a high level
using solely the functions of the system.

Further information about this system (data, more detailed models) is presented
at: https://c3.nasa.gov/dashlink/projects/3/
"""
from fmdtools.define.block import FxnBlock, Mode
from fmdtools.define.model import Model
from fmdtools.define.parameter import Parameter, SimParam
from fmdtools.define.state import State
from fmdtools.define.flow import Flow


class GenericState(State):
    rate: float = 1.0
    effort: float = 1.0


class GenericFlow(Flow):
    _init_s = GenericState


class SigState(State):
    value: float = 1.0


class Signal(Flow):
    _init_s = SigState


class ImportEEModes(Mode):
    faultparams = {"low_v": (1e-5, 100), "high_v": (5e-6, 100), "no_v": (1e-5, 300)}


class ImportEE(FxnBlock):
    __slots__ = ("ee_out",)
    _init_m = ImportEEModes
    _init_ee_out = GenericFlow
    flownames = {"ee_1": "ee_out"}
    """ Static model representation is the same as the dynamic model respresentation, except in this case 
    there is no opportunity vector. Thus the self.assoc_modes function takes a dictionary of modes with 
    just the vector of failure distribution and results cost. e.g. {'modename':[rate, cost]}.
    
    Also note that this model sets up the probability model differently--instead of specifying an overall failure rate
    for the function, one instead specifies an individual rate for eaach mode.
    
    Both representations can be used--this just shows this representation.
    """

    def behavior(self, time):
        if self.m.has_fault("no_v"):
            self.ee_out.s.effort = 0.0
        elif self.m.has_fault("high_v"):
            self.ee_out.s.effort = 2.0
        elif self.m.has_fault("low_v"):
            self.ee_out.s.effort = 0.5
        else:
            self.ee_out.s.effort = 1.0


class ImportSigModes(Mode):
    faultparams = {"partial_signal": (1e-5, 750), "no_signal": (1e-6, 750)}


class ImportSig(FxnBlock):
    __slots__ = ("sig_out",)
    _init_m = ImportSigModes
    _init_sig_out = Signal
    flownames = {"sig_in": "sig_out"}

    def behavior(self, time):
        if self.m.has_fault("partial_signal"):
            self.sig_out.s.value = 0.5
        elif self.m.has_fault("no_signal"):
            self.sig_out.s.value = 0.0
        else:
            self.sig_out.s.value = 1.0


class StoreEEModes(Mode):
    faultparams = {"low_storage": (5e-6, 2000), "no_storage": (5e-6, 2000)}


class StoreEE(FxnBlock):
    __slots__ = ("ee_in", "ee_out")
    _init_m = StoreEEModes
    _init_ee_in = GenericFlow
    _init_ee_out = GenericFlow
    flownames = {"ee_2": "ee_in", "ee_3": "ee_out"}

    def condfaults(self, time):
        if self.ee_out.s.effort * self.ee_out.s.rate >= 4.0:
            self.m.add_fault("no_storage")
        elif self.ee_out.s.effort * self.ee_out.s.rate >= 2.0:
            self.m.add_fault("low_storage")

    def behavior(self, time):
        if self.m.has_fault("no_storage"):
            self.ee_out.s.effort = 0.0
            self.ee_in.s.rate = 1.0
        elif self.m.has_fault("low_storage"):
            self.ee_out.s.effort = 1.0
            self.ee_in.s.rate = self.ee_out.s.rate
        else:
            self.ee_out.s.effort = self.ee_in.s.effort
            self.ee_in.s.rate = self.ee_out.s.rate


class SupplyEEModes(Mode):
    faultparams = {
        "adverse_resist": (2e-6, 400),
        "minor_overload": (1e-5, 400),
        "major_overload": (3e-6, 400),
        "short": (1e-7, 400),
        "open_circuit": (5e-8, 200),
    }


class SupplyEE(FxnBlock):
    __slots__ = ("ee_in", "ee_out", "heat_out")
    _init_m = SupplyEEModes
    _init_ee_in = GenericFlow
    _init_ee_out = GenericFlow
    _init_heat_out = GenericFlow
    flownames = {"ee_1": "ee_in", "ee_2": "ee_out", "waste_he_1": "heat_out"}

    def condfaults(self, time):
        if self.ee_out.s.rate > 2.0:
            self.m.add_fault("short")
        elif self.ee_out.s.rate > 1.0:
            self.m.add_fault("open_circuit")

    def behavior(self, time):
        if self.m.has_fault("open_circuit"):
            self.ee_out.s.effort = 0.0
            self.ee_in.s.rate = 1.0
        elif self.m.has_fault("short"):
            self.ee_out.s.effort = self.ee_in.s.effort * 4.0
            self.ee_in.s.rate = 4.0
        elif self.m.has_fault("major_overload"):
            self.ee_out.s.effort = self.ee_in.s.effort + 1.0
            self.heat_out.s.effort = 2.0
        elif self.m.has_fault("minor_overload"):
            self.ee_out.s.effort = 4.0
            self.heat_out.s.effort = 4.0
        elif self.m.has_fault("adverse_resist"):
            self.ee_out.s.effort = self.ee_in.s.effort - 1.0
        else:
            self.ee_out.s.effort = self.ee_in.s.effort
            self.heat_out.s.effort = 1.0


class DistEEModes(Mode):
    faultparams = {
        "adverse_resist": (1e-5, 1500),
        "poor_alloc": (2e-5, 500),
        "short": (2e-5, 1500),
        "open_circuit": (3e-5, 1500),
    }


class DistEE(FxnBlock):
    __slots__ = ("sig_in", "ee_in", "ee_m", "ee_h", "ee_o")
    _init_m = DistEEModes
    _init_sig_in = GenericFlow
    _init_ee_in = GenericFlow
    _init_ee_m = GenericFlow
    _init_ee_h = GenericFlow
    _init_ee_o = GenericFlow
    flownames = {"ee_3": "ee_in"}

    def condfaults(self, time):
        if max(self.ee_m.s.rate, self.ee_h.s.rate, self.ee_o.s.rate) > 2.0:
            self.m.add_fault("short")

    def behavior(self, time):
        if self.m.has_fault("short"):
            self.ee_in.s.rate = self.ee_in.s.effort * 4.0
            self.ee_m.s.effort = 0.0
            self.ee_h.s.effort = 0.0
            self.ee_o.s.effort = 0.0
        elif self.m.has_fault("open_circuit") or self.sig_in.s.value <= 0.0:
            self.ee_in.s.rate = 0.0
            self.ee_m.s.effort = 0.0
            self.ee_h.s.effort = 0.0
            self.ee_o.s.effort = 0.0
        elif (
            self.m.has_fault("poor_alloc")
            or self.m.has_fault("adverse_resist")
            or self.sig_in.s.value < 1.0
        ):
            self.ee_m.s.effort = self.ee_in.s.effort - 1.0
            self.ee_h.s.effort = self.ee_in.s.effort - 1.0
            self.ee_o.s.effort = self.ee_in.s.effort - 1.0
            self.ee_in.s.rate = max(
                self.ee_m.s.rate, self.ee_h.s.rate, self.ee_o.s.rate
            )
        else:
            self.ee_m.s.effort = self.ee_in.s.effort
            self.ee_h.s.effort = self.ee_in.s.effort
            self.ee_o.s.effort = self.ee_in.s.effort
            self.ee_in.s.rate = max(
                self.ee_m.s.rate, self.ee_h.s.rate, self.ee_o.s.rate
            )


class ExportHEModes(Mode):
    faultparams = {"hot_sink": (1e-5, 500), "ineffective_sink": (0.5e-5, 1000)}


class ExportHE(FxnBlock):
    __slots__ = ("he",)
    _init_m = ExportHEModes
    _init_he = GenericFlow
    flownames = {"waste_he_1": "he", "waste_he_o": "he", "waste_he_m": "he"}

    def behavior(self, time):
        if self.m.has_fault("ineffective_sink"):
            self.he.s.rate = 4.0
        elif self.m.has_fault("hot_sink"):
            self.he.s.rate = 2.0
        else:
            self.he.s.rate = 1.0


class ExportME(FxnBlock):
    __slots__ = ("me",)
    _init_me = GenericFlow

    def behavior(self, time):
        self.me.s.rate = self.me.s.effort


class ExportOE(FxnBlock):
    __slots__ = ("oe",)
    _init_oe = GenericFlow

    def behavior(self, time):
        self.oe.s.rate = self.oe.s.effort


class EEtoMEModes(Mode):
    faultparams = {
        "high_torque": (1e-4, 200),
        "low_torque": (1e-4, 200),
        "toohigh_torque": (5e-5, 200),
        "open_circuit": (5e-5, 200),
        "short": (5e-5, 200),
    }


class EEtoME(FxnBlock):
    __slots__ = ("ee_in", "me", "he_out")
    _init_m = EEtoMEModes
    _init_ee_in = GenericFlow
    _init_me = GenericFlow
    _init_he_out = GenericFlow
    flownames = {"ee_m": "ee_in", "waste_he_m": "he_out"}

    def behavior(self, time):
        if self.m.has_fault("high_torque"):
            self.he_out.s.effort = self.ee_in.s.effort + 1.0
            self.me.s.effort = self.ee_in.s.effort + 1.0
            self.ee_in.s.rate = 1.0 / (self.me.s.rate + 0.001) - 1.0
        elif self.m.has_fault("low_torque"):
            self.he_out.s.effort = self.ee_in.s.effort - 1.0
            self.me.s.effort = self.ee_in.s.effort - 1.0
            self.ee_in.s.rate = 1.0 / (self.me.s.rate + 0.001) - 1.0
        elif self.m.has_fault("toohigh_torque"):
            self.he_out.s.effort = 4.0
            self.me.s.effort = 4.0
            self.ee_in.s.rate = 4.0
        elif self.m.has_fault("open_circuit"):
            self.he_out.s.effort = 0.0
            self.me.s.effort = 0.0
            self.me.s.rate = 0.0
            self.ee_in.s.rate = 0.0
        elif self.m.has_fault("short"):
            self.ee_in.s.rate = self.ee_in.s.effort * 4.0
            self.he_out.s.effort = self.ee_in.s.effort
            self.me.s.effort = 0.0
            self.me.s.rate = 0.0
        else:
            self.he_out.s.effort = self.ee_in.s.effort
            self.me.s.effort = self.ee_in.s.effort
            self.me.s.rate = self.ee_in.s.effort
            self.ee_in.s.rate = self.ee_in.s.effort


class EEtoHEModes(Mode):
    faultparams = {
        "low_heat": (2e-6, 200),
        "high_heat": (1e-7, 200),
        "toohigh_heat": (5e-7, 200),
        "open_circuit": (1e-7, 200),
    }


class EEtoHE(FxnBlock):
    __slots__ = ("ee_in", "he")
    _init_m = EEtoHEModes
    _init_ee_in = GenericFlow
    _init_he = GenericFlow
    flownames = {"ee_h": "ee_in"}

    def cond_faults(self, time):
        if self.ee_in.s.effort > 2.0:
            self.m.add_fault("open_circuit")
        elif self.ee_in.s.effort > 1.0:
            self.m.add_fault("low_heat")

    def behavior(self, time):
        if self.m.has_fault("open_circuit"):
            self.he.s.effort = 0.0
            self.ee_in.s.rate = 0.0
        elif self.m.has_fault("low_heat"):
            self.he.s.effort = self.ee_in.s.effort - 1.0
            self.ee_in.s.rate = self.ee_in.s.effort
        elif self.m.has_fault("high_heat"):
            self.he.s.effort = self.ee_in.s.effort + 1.0
            self.ee_in.s.rate = self.ee_in.s.effort + 1.0
        elif self.m.has_fault("toohigh_heat"):
            self.he.s.effort = 4.0
            self.ee_in.s.rate = 4.0
        else:
            self.he.s.effort = self.ee_in.s.effort
            self.ee_in.s.rate = self.ee_in.s.effort


class EEtoOEModes(Mode):
    faultparams = {"optical_resist": (5e-7, 70), "burnt_out": (2e-6, 100)}


class EEtoOE(FxnBlock):
    __slots__ = ("ee_in", "oe", "he_out")
    _init_m = EEtoOEModes
    _init_ee_in = GenericFlow
    _init_oe = GenericFlow
    _init_he_out = GenericFlow
    flownames = {"waste_he_o": "he_out", "ee_o": "ee_in"}

    def cond_faults(self, time):
        if self.ee_in.s.effort >= 2.0:
            self.m.add_fault("burnt_out")

    def behavior(self, time):
        if self.m.has_fault("burnt_out"):
            self.ee_in.s.rate = 0.0
            self.he_out.s.effort = 0.0
            self.oe.s.effort = 0.0
        elif self.m.has_fault("optical_resist"):
            self.ee_in.s.rate = self.ee_in.s.effort - 1.0
            self.he_out.s.effort = self.ee_in.s.effort - 1.0
            self.oe.s.effort = self.ee_in.s.effort - 1.0
        else:
            self.ee_in.s.rate = self.ee_in.s.effort
            self.he_out.s.effort = self.ee_in.s.effort
            self.oe.s.effort = self.ee_in.s.effort


class EPS(Model):
    __slots__ = ()
    default_track = {"flows": ["he", "me", "oe"]}

    def __init__(self, sp=SimParam(times=(0, 1)), **kwargs):
        """
        The Model superclass uses a static model representation by default if
        there are no parameters for times, phases, etc.
        """
        super().__init__(sp=sp, **kwargs)

        self.add_flow("ee_1", GenericFlow)
        self.add_flow("ee_2", GenericFlow)
        self.add_flow("ee_3", GenericFlow)
        self.add_flow("ee_m", GenericFlow)
        self.add_flow("ee_o", GenericFlow)
        self.add_flow("ee_h", GenericFlow)
        self.add_flow("me", GenericFlow)
        self.add_flow("oe", GenericFlow)
        self.add_flow("he", GenericFlow)
        self.add_flow("waste_he_1", GenericFlow)
        self.add_flow("waste_he_o", GenericFlow)
        self.add_flow("waste_he_m", GenericFlow)
        self.add_flow("sig_in", Signal)

        self.add_fxn("import_ee", ImportEE, "ee_1")
        self.add_fxn("supply_ee", SupplyEE, "ee_1", "ee_2", "waste_he_1")
        self.add_fxn("store_ee", StoreEE, "ee_2", "ee_3")
        self.add_fxn("import_signal", ImportSig, "sig_in")
        self.add_fxn("distribute_ee", DistEE, "sig_in", "ee_3", "ee_m", "ee_h", "ee_o")
        self.add_fxn("ee_to_me", EEtoME, "ee_m", "me", "waste_he_m")
        self.add_fxn("ee_to_oe", EEtoOE, "ee_o", "oe", "waste_he_o")
        self.add_fxn("ee_to_he", EEtoHE, "ee_h", "he")
        self.add_fxn("export_me", ExportME, "me")
        self.add_fxn("export_he", ExportHE, "he")
        self.add_fxn("export_oe", ExportOE, "oe")
        self.add_fxn("export_waste_h1", ExportHE, "waste_he_1")
        self.add_fxn("export_waste_ho", ExportHE, "waste_he_o")
        self.add_fxn("export_waste_hm", ExportHE, "waste_he_m")

        self.build()

    def find_classification(self, scen, mdlhists):
        outflows = ["he", "me", "oe"]

        qualfunc = [
            [-90.0, -80.0, -70.0, -85.0, -100.0],
            [-80.0, -50.0, -20, -15, -100.0],
            [-70.0, -20.0, 0.0, -20.0, -100.0],
            [-85.0, -10, -20.0, -50.0, -110.0],
            [-100.0, -100.0, -100.0, -110.0, -110.0],
        ]

        flowcost = -5 * sum(
            [
                qualfunc[discrep(self.flows[fl].s.effort)][
                    discrep(self.flows[fl].s.rate)
                ]
                for fl in outflows
            ]
        )

        repcost = self.calc_repaircost()
        cost = repcost + flowcost

        rate = scen.rate
        return {"rate": rate, "cost": cost, "expected cost": 24 * 365 * 5 * rate * cost}


def discrep(value):
    if value <= 0.0:
        return 0
    elif value <= 0.5:
        return 1
    elif value <= 1.0:
        return 2
    elif value <= 2.0:
        return 3
    else:
        return 4


if __name__ == "__main__":
    import fmdtools.sim.propagate as propagate
    from fmdtools.analyze.graph import ModelGraph
    import numpy as np

    mdl = EPS()

    result, mdlhists = propagate.one_fault(
        mdl, "distribute_ee", "short", desired_result="graph", track="all"
    )

    result.graph.draw()
    # endclasses, mdlhists = propagate.single_faults(mdl)

    # resgraph, mdlhists = propagate.one_fault(mdl, 'ee_to_me', 'toohigh_torque', desired_result="fxnflowgraph")
    # result.graph.draw()

    summary = mdlhists.get_fault_degradation_summary(*mdl.fxns, *mdl.flows)
    # endclasses, mdlhists = propagate.single_faults(mdl)
    degradation = mdlhists.get_degraded_hist(*mdl.fxns, *mdl.flows)

    degtimemap = degradation.get_summary(operator=np.sum)

    mg = ModelGraph(mdl)
    mg.set_heatmap(degtimemap)
    mg.draw()

    propagate.single_faults(mdl)

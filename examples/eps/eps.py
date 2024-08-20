#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EPS Model.

This electrical power system model showcases how fmdtools can be used for purely static
propogation models (where the dynamic states are not a concern). This EPS system was
previously provided in the IBFM fault modelling toolkit
(see: https://github.com/DesignEngrLab/IBFM ) and other references--this implementation
follows the simple_eps model in IBFM.

The main purpose of this system is to supply electrical power to optical, mechanical,
and heat loads. In this model, we represent the failure behavior of the system at a high
level using solely the functions of the system.

Further information about this system (data, more detailed models) is presented
at: https://c3.nasa.gov/dashlink/projects/3/

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

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow


class GenericState(State):
    rate: float = 1.0
    effort: float = 1.0


class GenericFlow(Flow):

    __slots__ = ()
    container_s = GenericState


class SigState(State):
    value: float = 1.0


class Signal(Flow):
    __slots__ = ()
    container_s = SigState


class ImportEEModes(Mode):
    fm_args = {"low_v": (1e-5, 100),
               "high_v": (5e-6, 100),
               "no_v": (1e-5, 300)}


class ImportEE(Function):

    __slots__ = ("ee_out",)
    container_m = ImportEEModes
    flow_ee_out = GenericFlow
    flownames = {"ee_1": "ee_out"}
    """
    ImportEE in static modelling case.

    Static model representation is the same as the dynamic model respresentation, except
    in this case there is no opportunity vector. Thus the self.assoc_modes function
    takes a dictionary of modes with just the vector of failure distribution and results
    cost. e.g. {'modename':[rate, cost]}.

    Also note that this model sets up the probability model differently--instead of
    specifying an overall failure rate for the function, one instead specifies an
    individual rate for eaach mode.

    Both representations can be used--this just shows this representation.
    """

    def static_behavior(self, time):
        if self.m.has_fault("no_v"):
            self.ee_out.s.effort = 0.0
        elif self.m.has_fault("high_v"):
            self.ee_out.s.effort = 2.0
        elif self.m.has_fault("low_v"):
            self.ee_out.s.effort = 0.5
        else:
            self.ee_out.s.effort = 1.0


class ImportSigModes(Mode):
    fm_args = {"partial_signal": (1e-5, 750),
               "no_signal": (1e-6, 750)}


class ImportSig(Function):

    __slots__ = ("sig_out",)
    container_m = ImportSigModes
    flow_sig_out = Signal
    flownames = {"sig_in": "sig_out"}

    def static_behavior(self, time):
        if self.m.has_fault("partial_signal"):
            self.sig_out.s.value = 0.5
        elif self.m.has_fault("no_signal"):
            self.sig_out.s.value = 0.0
        else:
            self.sig_out.s.value = 1.0


class StoreEEModes(Mode):
    fm_args = {"low_storage": (5e-6, 2000),
               "no_storage": (5e-6, 2000)}


class StoreEE(Function):

    __slots__ = ("ee_in", "ee_out")
    container_m = StoreEEModes
    flow_ee_in = GenericFlow
    flow_ee_out = GenericFlow
    flownames = {"ee_2": "ee_in", "ee_3": "ee_out"}

    def set_faults(self):
        if self.ee_out.s.effort * self.ee_out.s.rate >= 4.0:
            self.m.add_fault("no_storage")
        elif self.ee_out.s.effort * self.ee_out.s.rate >= 2.0:
            self.m.add_fault("low_storage")

    def static_behavior(self, time):
        self.set_faults()
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
    fm_args = {"adverse_resist": (2e-6, 400),
               "minor_overload": (1e-5, 400),
               "major_overload": (3e-6, 400),
               "short": (1e-7, 400),
               "open_circuit": (5e-8, 200)}


class SupplyEE(Function):

    __slots__ = ("ee_in", "ee_out", "heat_out")
    container_m = SupplyEEModes
    flow_ee_in = GenericFlow
    flow_ee_out = GenericFlow
    flow_heat_out = GenericFlow
    flownames = {"ee_1": "ee_in", "ee_2": "ee_out", "waste_he_1": "heat_out"}

    def set_faults(self):
        if self.ee_out.s.rate > 2.0:
            self.m.add_fault("short")
        elif self.ee_out.s.rate > 1.0:
            self.m.add_fault("open_circuit")

    def static_behavior(self, time):
        self.set_faults()
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
    fm_args = {"adverse_resist": (1e-5, 1500),
               "poor_alloc": (2e-5, 500),
               "short": (2e-5, 1500),
               "open_circuit": (3e-5, 1500)}


class DistEE(Function):

    __slots__ = ("sig_in", "ee_in", "ee_m", "ee_h", "ee_o")
    container_m = DistEEModes
    flow_sig_in = Signal
    flow_ee_in = GenericFlow
    flow_ee_m = GenericFlow
    flow_ee_h = GenericFlow
    flow_ee_o = GenericFlow
    flownames = {"ee_3": "ee_in"}

    def set_faults(self):
        if max(self.ee_m.s.rate, self.ee_h.s.rate, self.ee_o.s.rate) > 2.0:
            self.m.add_fault("short")

    def static_behavior(self, time):
        self.set_faults()
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
    fm_args = {"hot_sink": (1e-5, 500),
               "ineffective_sink": (0.5e-5, 1000)}


class ExportHE(Function):

    __slots__ = ("he",)
    container_m = ExportHEModes
    flow_he = GenericFlow
    flownames = {"waste_he_1": "he", "waste_he_o": "he", "waste_he_m": "he"}

    def static_behavior(self, time):
        if self.m.has_fault("ineffective_sink"):
            self.he.s.rate = 4.0
        elif self.m.has_fault("hot_sink"):
            self.he.s.rate = 2.0
        else:
            self.he.s.rate = 1.0


class ExportME(Function):

    __slots__ = ("me",)
    flow_me = GenericFlow

    def static_behavior(self, time):
        self.me.s.rate = self.me.s.effort


class ExportOE(Function):

    __slots__ = ("oe",)
    flow_oe = GenericFlow

    def static_behavior(self, time):
        self.oe.s.rate = self.oe.s.effort


class EEtoMEModes(Mode):
    fm_args = {"high_torque": (1e-4, 200),
               "low_torque": (1e-4, 200),
               "toohigh_torque": (5e-5, 200),
               "open_circuit": (5e-5, 200),
               "short": (5e-5, 200)}


class EEtoME(Function):

    __slots__ = ("ee_in", "me", "he_out")
    container_m = EEtoMEModes
    flow_ee_in = GenericFlow
    flow_me = GenericFlow
    flow_he_out = GenericFlow
    flownames = {"ee_m": "ee_in", "waste_he_m": "he_out"}

    def static_behavior(self, time):
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
    fm_args = {"low_heat": (2e-6, 200),
               "high_heat": (1e-7, 200),
               "toohigh_heat": (5e-7, 200),
               "open_circuit": (1e-7, 200)}


class EEtoHE(Function):

    __slots__ = ("ee_in", "he")
    container_m = EEtoHEModes
    flow_ee_in = GenericFlow
    flow_he = GenericFlow
    flownames = {"ee_h": "ee_in"}

    def cond_faults(self, time):
        if self.ee_in.s.effort > 2.0:
            self.m.add_fault("open_circuit")
        elif self.ee_in.s.effort > 1.0:
            self.m.add_fault("low_heat")

    def static_behavior(self, time):
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
    fm_args = {"optical_resist": (5e-7, 70),
               "burnt_out": (2e-6, 100)}


class EEtoOE(Function):

    __slots__ = ("ee_in", "oe", "he_out")
    container_m = EEtoOEModes
    flow_ee_in = GenericFlow
    flow_oe = GenericFlow
    flow_he_out = GenericFlow
    flownames = {"waste_he_o": "he_out", "ee_o": "ee_in"}

    def cond_faults(self, time):
        if self.ee_in.s.effort >= 2.0:
            self.m.add_fault("burnt_out")

    def static_behavior(self, time):
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


class EPS(FunctionArchitecture):

    __slots__ = ()
    default_track = {"flows": ["he", "me", "oe"]}
    default_sp = {'end_time': 0}

    def init_architecture(self, **kwargs):
        """
        The Model superclass uses a static model representation by default if
        there are no parameters for times, phases, etc.
        """
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
        return {"rate": rate, "cost": cost, "expected_cost": 24 * 365 * 5 * rate * cost}


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
    from fmdtools.define.architecture.function import FunctionArchitectureGraph
    import numpy as np

    mdl = EPS(track=['fxns', 'flows', 'i'])
    result, mdlhists = propagate.one_fault(mdl, "distribute_ee", "short")

    result, mdlhists = propagate.one_fault(mdl, "distribute_ee", "short",
                                           desired_result="graph")

    result.graph.draw()
    # endclasses, mdlhists = propagate.single_faults(mdl)

    # resgraph, mdlhists = propagate.one_fault(mdl, 'ee_to_me', 'toohigh_torque', desired_result="fxnflowgraph")
    # result.graph.draw()
    ks = [*mdl.get_roles_as_dict("fxn", "flow", flex_prefixes=True)]

    summary = mdlhists.get_fault_degradation_summary(*ks)
    # endclasses, mdlhists = propagate.single_faults(mdl)
    degradation = mdlhists.get_degraded_hist(*ks)

    degtimemap = degradation.get_summary(operator=np.sum)

    mg = FunctionArchitectureGraph(mdl)
    mg.set_heatmap({'eps.'+k: v for k, v in degtimemap.items()})
    mg.draw()

    #propagate.single_faults(mdl)

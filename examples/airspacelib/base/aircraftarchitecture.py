# -*- coding: utf-8 -*-
"""
Combined aircraft architecture with all of the subfunctions in /arch.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.architecture.function import FunctionArchitectureGraph
from fmdtools.define.container.parameter import Parameter
import fmdtools.sim.propagate as prop


from examples.airspacelib.base.arch.flows import Trajectories, Force, Electricity
from examples.airspacelib.base.arch.flows import AircraftEnvironment
from examples.airspacelib.base.arch.aviate import Aviate
from examples.airspacelib.base.arch.controlflight import ControlFlight
from examples.airspacelib.base.arch.storeee import StoreAndSupplyElectricity
from examples.airspacelib.base.arch.perceiveenvironment import PerceiveEnvironment
from examples.airspacelib.base.arch.holdpayload import HoldPayload


class AircraftArchParameter(Parameter):
    """Overall Parameter Defining the AircraftArchitecture."""

    flightplan: tuple = ((0.0, 0.0), (25.0, 25.0))
    height: float = 25.0


class AircraftArchitecture(FunctionArchitecture):
    """
    Overall drone architecture.

    Involves flows:
        - force: the flow of force between physical components
        - electricity: the flow of electrical energy from the power supply to functions
        - trajectories: the 3d position and velocity of the aircraft
        - environment: the environment the aircraft inhabits and interacts with
    And functions:
        - control_flight : flight planning and control
        - aviate : the function that moves the drone in the x/y/z
        - store_and_supply_ee : the aircraft power supply/battery
        - perceive_environment : drone perception and localization
        - hold_payload : the structure of the drone
    """

    __slots__ = ()
    container_p = AircraftArchParameter

    def init_architecture(self, **kwargs):
        """Initialize the architecture of the aircraft."""
        self.add_flow('force', Force)
        self.add_flow('electricity', Electricity)
        self.add_flow('trajectories', Trajectories)
        self.add_flow('environment', AircraftEnvironment)

        self.add_fxn('control_flight', ControlFlight,
                     'trajectories', 'force', 'electricity', 'environment',
                     s={'flightplan': self.p.flightplan, 'height': self.p.height})
        self.add_fxn('aviate', Aviate,
                     'trajectories', 'force', 'electricity', 'environment')
        self.add_fxn('store_and_supply_ee', StoreAndSupplyElectricity,
                     'force', 'electricity')
        self.add_fxn('perceive_environment', PerceiveEnvironment,
                     'environment', 'force', 'electricity', 'trajectories')
        self.add_fxn('hold_payload', HoldPayload, 'trajectories', 'force')

    def classify(self, **kwargs):
        """Classify the simulation results."""
        return {'faultmodes': {*self.return_faultmodes()}}


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    t = Trajectories()
    t.create_local('perc_traj')
    cf = ControlFlight(trajectories=t)
    av = Aviate(trajectories=t)

    cf.dynamic_behavior()
    cf.des_traj.s
    av.dynamic_behavior()
    av.trajectories.s

    cf.dynamic_behavior()
    cf.des_traj.s
    av.dynamic_behavior()
    av.trajectories.s

    da = AircraftArchitecture(p={'flightplan':((0.0, 0.0), (25.0, 0.0), (0.0, 25.0), (25.0, 25.0))})
    fg = FunctionArchitectureGraph(da)
    fg.draw()

    res, hist = prop.nominal(da)

    res, hist = prop.one_fault(da, 'store_and_supply_ee', 'depletion', 7, to_return=['classify', 'graph'])
    res.store_and_supply_ee_depletion_t7.tend.graph.draw()

    hist.plot_trajectories('trajectories.s.x', 'trajectories.s.y')
    hist.plot_trajectories('trajectories.s.x',
                           'trajectories.s.y',
                           'trajectories.s.z',
                           time_groups='nominal', time_ticks=1.0)

    hist.plot_line('flows.electricity.s.charge',
                   'fxns.control_flight.m.mode',
                   'fxns.aviate.m.mode')

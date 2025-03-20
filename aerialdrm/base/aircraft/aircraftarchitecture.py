# -*- coding: utf-8 -*-
"""Combined aircraft architecture with all of the subfunctions in /arch."""
from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.architecture.function import FunctionArchitectureGraph
from fmdtools.define.container.parameter import Parameter
import fmdtools.sim.propagate as prop


from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity
from aerialdrm.base.aircraft.arch.flows import Environment
from aerialdrm.base.aircraft.arch.aviate import Aviate
from aerialdrm.base.aircraft.arch.controlflight import ControlFlight
from aerialdrm.base.aircraft.arch.storeee import StoreAndSupplyElectricity
from aerialdrm.base.aircraft.arch.perceiveenvironment import PerceiveEnvironment
from aerialdrm.base.aircraft.arch.holdpayload import HoldPayload


class AircraftArchParameter(Parameter):

    flightplan: tuple = ((0.0, 0.0), (25.0, 25.0))
    height: float = 25.0


class AircraftArchitecture(FunctionArchitecture):
    __slots__ = ()
    container_p = AircraftArchParameter

    def init_architecture(self, **kwargs):
        self.add_flow('force', Force)
        self.add_flow('electricity', Electricity)
        self.add_flow('trajectories', Trajectories)
        self.add_flow('environment', Environment)

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

    def find_classification(self, scen, mdlhists):
        return {'faultmodes': {*self.return_faultmodes()}}


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    t = Trajectories()
    t.create_local('perc_traj')
    cf = ControlFlight(trajectories=t)
    av = Aviate(trajectories=t)

    cf.dynamic_behavior(1)
    cf.des_traj.s
    av.dynamic_behavior(1)
    av.trajectories.s

    cf.dynamic_behavior(2)
    cf.des_traj.s
    av.dynamic_behavior(2)
    av.trajectories.s

    da = AircraftArchitecture(p={'flightplan':((0.0, 0.0), (25.0, 0.0), (0.0, 25.0), (25.0, 25.0))})
    fg = FunctionArchitectureGraph(da)
    fg.draw()

    res, hist = prop.nominal(da)

    res, hist = prop.one_fault(da, 'store_and_supply_ee', 'break', 7, desired_result=['endclass', 'graph'])
    res.graph.draw()

    hist.plot_trajectories('trajectories.s.x', 'trajectories.s.y')
    hist.plot_trajectories('trajectories.s.x',
                           'trajectories.s.y',
                           'trajectories.s.z',
                           time_groups='nominal', time_ticks=1.0)

    hist.plot_line('fxns.store_and_supply_ee.s.charge',
                   'fxns.control_flight.m.mode',
                   'fxns.aviate.m.mode')

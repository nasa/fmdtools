# -*- coding: utf-8 -*-
"""Combined aircraft architecture with all of the subfunctions in /arch."""
from fmdtools.define.architecture.function import FunctionArchitecture, FunctionArchitectureGraph
import fmdtools.sim.propagate as prop


from aerialdrm.base.aircraft.arch.flows import Trajectories, Force, Electricity
from aerialdrm.base.aircraft.arch.flows import Environment
from aerialdrm.base.aircraft.arch.aviate import Aviate
from aerialdrm.base.aircraft.arch.controlflight import ControlFlight
from aerialdrm.base.aircraft.arch.storeee import StoreAndSupplyElectricity
from aerialdrm.base.aircraft.arch.perceiveenvironment import PerceiveEnvironment


class AircraftArchitecture(FunctionArchitecture):
    __slots__ = ()

    def init_architecture(self, **kwargs):
        self.add_flow('force', Force)
        self.add_flow('electricity', Electricity)
        self.add_flow('trajectories', Trajectories)
        self.add_flow('environment', Environment)

        self.add_fxn('control_flight', ControlFlight,
                     'trajectories', 'force', 'electricity', 'environment')
        self.add_fxn('aviate', Aviate,
                     'trajectories', 'force', 'electricity', 'environment')
        self.add_fxn('store_and_supply_ee', StoreAndSupplyElectricity,
                     'force', 'electricity')
        self.add_fxn('perceive_environment', PerceiveEnvironment,
                     'environment', 'force', 'electricity', 'trajectories')


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

    da = AircraftArchitecture()
    fg = FunctionArchitectureGraph(da)
    fg.draw()

    res, hist = prop.nominal(da)

    hist.plot_trajectories('trajectories.s.x', 'trajectories.s.y')
    hist.plot_trajectories('trajectories.s.x',
                           'trajectories.s.y',
                           'trajectories.s.z',
                           time_groups='nominal', time_ticks=1.0)

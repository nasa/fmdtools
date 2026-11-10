# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:19:10 2026

@author: dhulse
"""


from fmdtools.define.architecture.function import FunctionArchitecture
from fmdtools.define.container.parameter import Parameter
from fmdtools.sim import propagate


from model_controlflight import ControlFlight, ControlParameter
from model_common import ControlsFlow, AircraftDOFs, AircraftTrajectory, AircraftParameter
from model_aviate import Aviate
from model_environment import AirportEnvironment


class AirportParameter(Parameter):
    controls: ControlParameter = ControlParameter()
    aircraft: AircraftParameter = AircraftParameter()
    


class Airport(FunctionArchitecture):
    container_p = AirportParameter
    default_sp = {'end_time': 100}
    default_t = {'force_locals': False}

    def init_architecture(self, **kwargs):
        self.add_flow('environment', AirportEnvironment)
        self.add_flow('controls', ControlsFlow)
        self.add_flow('aircraft_dofs', AircraftDOFs)
        self.add_flow('aircraft_trajectory', AircraftTrajectory)

        self.add_fxn('aviate', Aviate, 'controls', 'aircraft_dofs', 'aircraft_trajectory', p=self.p.aircraft)
        self.add_fxn('control_flight', ControlFlight, 'controls', 'aircraft_dofs', 'aircraft_trajectory', 'environment', p=self.p.controls)


if __name__ == "__main__":
    """
    ap = Airport()
    ap(time='end')

    fig, ax = ap.flows['environment'].ga.show_infrastructure()
    fig, ax = ap.h.plot_trajectories('flows.aircraft_trajectory.s.x', 'flows.aircraft_trajectory.s.y', fig=fig, ax=ax, time_groups=['nominal'], time_ticks=20.0)

    ap.h.plot_trajectories('flows.aircraft_trajectory.s.x', 'flows.aircraft_trajectory.s.y','flows.aircraft_trajectory.s.z', time_groups=['nominal'], time_ticks=20.0)
    ap.h.plot_trajectories('flows.aircraft_dofs.s.xe', 'flows.aircraft_dofs.s.ye','flows.aircraft_dofs.s.ze', time_groups=['nominal'], time_ticks=20.0)

    ap.h.plot_line("fxns.control_flight.m.mode")

    fig, ax = ap.flows['environment'].ga.show_infrastructure()
    
    fig, ax = ap.flows['environment'].ga.show(figsize=(10,10), z=True, legend=False)

    fig, ax = ap.h.plot_trajectories('flows.aircraft_trajectory.s.x', 'flows.aircraft_trajectory.s.y', 'flows.aircraft_trajectory.s.z', fig=fig, ax=ax, time_groups=['nominal'], time_ticks=20.0, legend=False)
    """
    ap = Airport()
    result, hist = propagate.one_fault(ap, 'control_flight', "false_abort", time=40)

    fig, ax = ap.flows['environment'].ga.show_infrastructure(overlay_labels=False,
                                                             title="Moffett Field Landing/Approach",
                                                             legend=False)
    hist.nominal.plot_trajectories('flows.aircraft_trajectory.s.x',
                                   'flows.aircraft_trajectory.s.y',
                                   fig=fig, ax=ax, time_groups=['nominal'],
                                   time_ticks=20.0, 
                                   legend = dict(bbox_to_anchor = (0.65, 1)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    hist.plot_line('flows.aircraft_trajectory.s.z',
                   'flows.aircraft_trajectory.s.distance',
                   'fxns.control_flight.m.mode',
                   'flows.controls.s.elevator',
                   'flows.controls.s.throttle',
                   'flows.aircraft_trajectory.s.dz',
                   'flows.aircraft_trajectory.s.goal_z')

    # result, hist = propagate.nominal(ap)
    fig, ax = ap.flows['environment'].ga.show_as_one_color(figsize=(5,5), z=True, legend=True,)
    ax.view_init(elev=10, azim=45, roll=0)
    hist.plot_trajectories('flows.aircraft_trajectory.s.x',
                           'flows.aircraft_trajectory.s.y',
                           'flows.aircraft_trajectory.s.z',
                           fig=fig, ax=ax, time_groups=['nominal', "faulty"],
                           xlabel="East/West",
                           ylabel="North/south",
                           zlabel="Elevation above runway (meters)",
                           time_ticks=20.0,
                           title="False Abort at Time t=40.0",
                           legend = dict(bbox_to_anchor = (0.4,0.75)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])



    # hist.flows.aircraft_trajectory.s.z
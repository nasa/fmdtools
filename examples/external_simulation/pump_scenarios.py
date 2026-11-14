"""
Compare reservoir volumes when the pump stops at different times.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


def sample_pump_stops(model, *, staged=False):
    """Run nominal and stopped pumps with the supplied model's rate and seed.

    In the default three-second run, a pump that stops does so at one second
    in 25% of stops and at two seconds in 75%. The simulation time grid must
    include both injection times. Set run_stochastic on the model to vary
    the rate, with matching random draws in all three scenarios.

    Return the FaultSample, terminal results and scenario histories.
    """
    stop_times = (1.0, 2.0)
    if not set(stop_times).issubset(model.sp.get_timerange()):
        raise ValueError("The simulation time grid must include 1 and 2 seconds")

    domain = FaultDomain(model)
    domain.add_fault(model.name, "stopped")
    sample = FaultSample(domain, def_mdl_phasemap=False)
    sample.add_fault_times(stop_times, weights=(0.25, 0.75))
    results, histories = propagate.fault_sample(
        model, sample, staged=staged, include_nominal=True, showprogress=False
    )
    return sample, results, histories

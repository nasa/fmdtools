#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstrate packet-delivery risk with seeded satellite simulations and FMEA.

The single-fault catalogue and its rates are synthetic, not an exhaustive
mission probability model. See risk_analysis.rst for the weighting convention.

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

import math
from dataclasses import dataclass
from numbers import Integral, Real

from fmdtools_examples.planetary_observation.model_disturbances import (
    DisturbedObservationSatellite,
)
from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import FMEA
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


FAULTS = (
    ("capture", "failed"),
    ("capture", "biased"),
    ("process", "failed"),
    ("downlink", "failed"),
)
METRICS = ["shortfall", "delivery_failed", "delta_shortfall"]


@dataclass
class RiskAnalysis:
    """Inspectable outputs; results/histories are nested under seed_<integer>.

    mean_result averages terminal metrics over seeds, including the no-fault
    reference named nominal. fmea excludes that reference. Its average columns
    use conditional injection-time weights; expected columns use fault rates.
    """

    seeds: tuple
    required_packets: int
    sample: FaultSample
    results: Result
    histories: History
    mean_result: Result
    fmea: FMEA


def make_fault_sample(model, times=(1, 3), time_weights=(0.25, 0.75)):
    """Sample declared single faults; time_weights are conditional on each fault.

    Times must be distinct positive integer minutes inside the mission. Weights
    must be finite, nonnegative and sum to one. Zero-weight scenarios are kept
    for inspection but contribute no risk. No joint faults are generated.
    """
    times, time_weights = tuple(times), tuple(time_weights)
    if not times or len(times) != len(time_weights):
        raise ValueError("Provide equally sized, nonempty times and time_weights")
    if any(
        isinstance(t, bool)
        or not isinstance(t, Real)
        or not math.isfinite(t)
        or t != int(t)
        or not 0 < t <= model.sp.end_time
        for t in times
    ) or len(set(times)) != len(times):
        raise ValueError("Injection times must be distinct integer mission minutes")
    if any(
        isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w < 0
        for w in time_weights
    ) or not math.isclose(math.fsum(time_weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Injection-time weights must be nonnegative and sum to one")
    domain = FaultDomain(model)
    domain.add_faults(*FAULTS)
    sample = FaultSample(domain, def_mdl_phasemap=False)
    sample.add_fault_times(list(times), weights=list(time_weights))
    return sample


def run_risk_analysis(
    seeds=(0,),
    *,
    stochastic=False,
    staged=True,
    params=None,
    times=(1, 3),
    time_weights=(0.25, 0.75),
    required_packets=2,
):
    """Simulate a fixed fault catalogue, average seeds, and build a native FMEA.

    Parameters
    ----------
    seeds : iterable of int
        Distinct nonnegative seeds. Each has equal Monte Carlo weight 1/N.
        The same seed is used for the no-fault and injected-fault realizations.
    stochastic : bool
        Enable the existing noise, obscuration and contact-loss distributions.
        False gives the hand-checkable deterministic reference.
    staged : bool
        Reuse pre-injection states through propagate.fault_sample when True.
    params : dict or None
        DisturbanceParam overrides; the original dictionary is not changed.
    times, time_weights : sequences
        Injection times and their conditional masses; see make_fault_sample.
    required_packets : int
        Nonnegative delivery target at the end of the six-minute mission.
        shortfall=max(required_packets-downlinked, 0); delivery_failed is its
        nonzero indicator. delta_shortfall subtracts the paired no-fault value
        and is signed. It must not be interpreted as a probability.

    Returns
    -------
    RiskAnalysis
        Per-seed outputs, their mean, and FMEA. Random draws already follow the
        input distributions, so latent probability densities are never used as
        extra weights. Fault contributions do not include no-fault background
        losses or unmodelled multiple faults and are not total mission risk.
    """
    seeds = tuple(seeds)
    if (
        not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Provide distinct nonnegative integer seeds")
    if (
        isinstance(required_packets, bool)
        or not isinstance(required_packets, Integral)
        or required_packets < 0
    ):
        raise ValueError("required_packets must be a nonnegative integer")
    seeds = tuple(int(seed) for seed in seeds)
    required_packets = int(required_packets)
    times, time_weights = tuple(times), tuple(time_weights)
    template = DisturbedObservationSatellite(
        p={} if params is None else params,
        r={"seed": seeds[0]},
        sp={"run_stochastic": stochastic},
    )
    sample = make_fault_sample(template, times, time_weights)
    results, histories = Result(), History()
    for seed in seeds:
        model = template.new(r={"seed": seed})
        result, history = propagate.fault_sample(
            model, sample, staged=staged, include_nominal=True, showprogress=False
        )
        baseline = max(required_packets - result["nominal.tend.classify.downlinked"], 0)
        for name in ["nominal", *sample.scen_names()]:
            prefix = name + ".tend.classify."
            shortfall = max(required_packets - result[prefix + "downlinked"], 0)
            result[prefix + "shortfall"] = shortfall
            result[prefix + "delivery_failed"] = int(shortfall > 0)
            result[prefix + "delta_shortfall"] = shortfall - baseline
        results["seed_" + str(seed)] = result
        histories["seed_" + str(seed)] = history

    # Equal averages over seeds, then fault/time weighting exactly once.
    mean_result = results.get_expected(round_value=False)
    time_mass = dict(zip(times, time_weights))
    conditional_weights = {
        scenario.name: time_mass[scenario.time] for scenario in sample.scenarios()
    }
    fmea = FMEA(
        mean_result,
        sample,
        group_by=("obj", "fault"),
        sum_metric="scenario_rate",
        average_metric=METRICS,
        expected_metric=METRICS,
        weights=conditional_weights,
        rates="scenario_rate",
        round_value=False,
    )
    return RiskAnalysis(
        seeds, required_packets, sample, results, histories, mean_result, fmea
    )


if __name__ == "__main__":
    analysis = run_risk_analysis()
    print(analysis.fmea.as_table(sort_by="expected_shortfall"))
    print(
        "Fault-catalogue weighted packet shortfall:",
        sum(analysis.fmea["expected_shortfall"].values()),
    )

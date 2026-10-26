#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run image-classification fault scenarios and a native FMEA for the satellite.

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

from numbers import Integral

from fmdtools_examples.planetary_observation.model_classifier import (
    ClassifyingSatellite,
)
from fmdtools_examples.planetary_observation.script_risk_analysis import (
    make_fault_sample,
)
from fmdtools.analyze.tabulate import FMEA
from fmdtools.sim import propagate


CLASSIFICATION_METRICS = [
    "missed_targets",
    "false_positives",
    "classification_errors",
    "unclassified",
]


def run_classification_analysis(
    *,
    seed=0,
    stochastic=False,
    staged=True,
    params=None,
    times=(1, 3),
    time_weights=(0.25, 0.75),
):
    """Return (result, history, sample, fmea) for one paired disturbance realization.

    Parameters follow run_risk_analysis, with one nonnegative integer seed.
    ImageMissionParam accepts held-out image indices and target digit classes.
    The average columns condition on injection time; expected columns apply
    the synthetic fault rates exactly once. Nominal outputs remain separate.
    A noisy single-seed result is one realization, not a Monte Carlo estimate.
    Detection counts describe selection, not the identity of downlinked data.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    model = ClassifyingSatellite(
        p={} if params is None else params,
        r={"seed": int(seed)},
        sp={"run_stochastic": stochastic},
    )
    times, time_weights = tuple(times), tuple(time_weights)
    sample = make_fault_sample(model, times, time_weights)
    result, history = propagate.fault_sample(
        model, sample, staged=staged, include_nominal=True, showprogress=False
    )
    time_mass = dict(zip(times, time_weights))
    table = FMEA(
        result,
        sample,
        sum_metric="scenario_rate",
        average_metric=CLASSIFICATION_METRICS,
        expected_metric=CLASSIFICATION_METRICS,
        weights={
            scenario.name: time_mass[scenario.time] for scenario in sample.scenarios()
        },
        rates="scenario_rate",
        round_value=False,
    )
    return result, history, sample, table


if __name__ == "__main__":
    result, history, sample, table = run_classification_analysis()
    print("No-fault classification counts:")
    print(result.get("nominal"))
    print("Synthetic single-fault contributions:")
    print(table.as_table(sort_by="expected_missed_targets"))

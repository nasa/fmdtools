#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytical reference cases for risk calculations.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “Fault Model Design tools - fmdtools version 2” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import unittest

from fmdtools.analyze.result import Result
from fmdtools.analyze.tabulate import FMEA


class ReferenceRiskSample:
    """Minimal independent scenario-rate model for FMEA reference checks."""

    groups = {
        ("sensor", "bias"): ["sensor_small", "sensor_large"],
        ("power", "loss"): ["power_short", "power_open"],
    }
    rates = {
        "sensor_small": 0.10,
        "sensor_large": 0.20,
        "power_short": 0.05,
        "power_open": 0.02,
    }

    def get_scen_groups(self, *factors):
        return self.groups

    def get_scen_values(self, value):
        if value != "rate":
            raise KeyError(value)
        return self.rates


class TestReferenceRiskCalculations(unittest.TestCase):
    """Check public risk-analysis outputs against hand-computable references."""

    def test_expected_loss_matches_discrete_probability_reference(self):
        # E[L] = 0*0.70 + 10*0.10 + 100*0.15 + 1000*0.05 = 66.
        result = Result(
            {
                "nominal.loss": 0.0,
                "minor.loss": 10.0,
                "major.loss": 100.0,
                "catastrophic.loss": 1000.0,
                "nominal.probability": 0.70,
                "minor.probability": 0.10,
                "major.probability": 0.15,
                "catastrophic.probability": 0.05,
            }
        )

        self.assertAlmostEqual(
            result.get_metric("loss", "expected", rates="probability"),
            66.0,
        )

    def test_event_probability_matches_discrete_probability_reference(self):
        # P(hazardous) = P(major) + P(catastrophic) = 0.15 + 0.05 = 0.20.
        result = Result(
            {
                "nominal.hazardous": 0,
                "minor.hazardous": 0,
                "major.hazardous": 1,
                "catastrophic.hazardous": 1,
                "nominal.probability": 0.70,
                "minor.probability": 0.10,
                "major.probability": 0.15,
                "catastrophic.probability": 0.05,
            }
        )

        self.assertAlmostEqual(
            result.get_metric("hazardous", "rate", rates="probability"),
            0.20,
        )

    def test_fmea_group_risk_matches_independent_rate_calculation(self):
        # sensor: 10*0.10 + 20*0.20 = 5
        # power:  100*0.05 + 500*0.02 = 15
        result = Result(
            {
                "sensor_small.tend.classify.cost": 10.0,
                "sensor_large.tend.classify.cost": 20.0,
                "power_short.tend.classify.cost": 100.0,
                "power_open.tend.classify.cost": 500.0,
            }
        )
        sample = ReferenceRiskSample()

        fmea = FMEA(
            result,
            sample,
            expected_metric=["cost"],
            rates="scenario_rate",
        )

        self.assertAlmostEqual(
            fmea.data["expected_cost"][("sensor", "bias")],
            5.0,
        )
        self.assertAlmostEqual(
            fmea.data["expected_cost"][("power", "loss")],
            15.0,
        )
        self.assertAlmostEqual(sum(fmea.data["expected_cost"].values()), 20.0)


if __name__ == "__main__":
    unittest.main()

"""
Check delivered volume and stop risk for the calibrated pump.

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

import unittest
from importlib.util import find_spec

import numpy as np
from fmdtools_examples.external_simulation.calibrated_pump import CalibratedPump
from fmdtools_examples.external_simulation.pump_scenarios import sample_pump_stops

from fmdtools.analyze.tabulate import FMEA


@unittest.skipUnless(find_spec("sklearn"), "requires fmdtools[examples]")
class TestCalibratedPumpRisk(unittest.TestCase):
    def test_volume_shortfall_per_run(self):
        model = CalibratedPump()
        sample, result, history = sample_pump_stops(model)
        nominal = result["nominal.tend.classify.value"]
        self.assertAlmostEqual(nominal, 4.8)
        delivered_at_stop = {1.0: 0.0, 2.0: 1.6}
        for scenario in sample.scenarios():
            prefix = scenario.name + ".tend.classify."
            delivered = result[prefix + "value"]
            self.assertAlmostEqual(delivered, delivered_at_stop[scenario.time])
            self.assertAlmostEqual(delivered, history.get(scenario.name)["s.value"][-1])
            result[prefix + "shortfall"] = nominal - delivered

        fmea = FMEA(
            result,
            sample,
            expected_metric="shortfall",
            rates="scenario_rate",
            round_value=False,
        )
        # Per run: 0.0025 * 4.8 L lost early + 0.0075 * 3.2 L lost late.
        expected_loss = fmea["expected_shortfall"][(model.name, "stopped")]
        self.assertAlmostEqual(expected_loss, 0.036)

    def test_staged_and_repeated_runs_preserve_seeded_trajectories(self):
        model = CalibratedPump(r={"seed": 31}, sp={"run_stochastic": True})
        sample, expected_result, expected_history = sample_pump_stops(model)
        for staged in (True, False):
            with self.subTest(staged=staged):
                _, result, history = sample_pump_stops(model, staged=staged)
                self.assertEqual(dict(result), dict(expected_result))
                for name in ("nominal", *sample.scen_names()):
                    for field in ("s.value", "r.s.multiplier"):
                        key = name + "." + field
                        np.testing.assert_array_equal(
                            history[key], expected_history[key]
                        )

    def test_copy_and_reset_keep_the_calibration(self):
        model = CalibratedPump()
        model(time=1.0)
        branch = model.copy()
        branch(time=3.0)
        self.assertAlmostEqual(branch.s.value, 4.8)
        self.assertAlmostEqual(model.s.value, 1.6)
        model(time=3.0)
        model.reset()
        self.assertEqual(model.s.value, 0.0)
        model(time=3.0)
        np.testing.assert_allclose(model.h.flatten()["s.value"], [0.0, 1.6, 3.2, 4.8])

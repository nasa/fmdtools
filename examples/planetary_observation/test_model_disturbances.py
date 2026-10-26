#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference tests for satellite disturbances, fault injection and seeded replay.

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

import copy
import math
import pickle
import unittest

import numpy as np

from fmdtools_examples.planetary_observation.model_main import ObservationSatellite
from fmdtools_examples.planetary_observation.model_disturbances import (
    DisturbanceParam,
    DisturbedObservationSatellite,
)
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestSatelliteDisturbances(unittest.TestCase):
    def assert_same(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def assert_accounting(self, result, history, capacity=4):
        opportunities = history["fxns.schedule.s.opportunities"]
        acquired = history["fxns.capture.s.acquired"]
        processed = history["fxns.process.s.processed"]
        selected = history["fxns.process.s.selected"]
        sent = history["fxns.downlink.s.downlinked"]
        queued = history["flows.buffer.s.queued"]
        dropped = history["fxns.process.s.dropped"]
        np.testing.assert_array_equal(
            opportunities, acquired + history["fxns.capture.s.missed"]
        )
        np.testing.assert_array_equal(
            acquired, processed + history["fxns.process.s.unprocessed"]
        )
        np.testing.assert_array_equal(selected, sent + queued + dropped)
        self.assertTrue(np.all((0 <= queued) & (queued <= capacity)))
        self.assertTrue(np.all((0 <= selected) & (selected <= processed)))
        score = history["flows.observation.s.signal"]
        self.assertTrue(np.all((0 <= score) & (score <= 1)))
        for name, values in (
            ("opportunities", opportunities),
            ("acquired", acquired),
            ("processed", processed),
            ("selected", selected),
            ("downlinked", sent),
            ("queued", queued),
            ("dropped", dropped),
        ):
            self.assertEqual(result["tend.classify." + name], values[-1])

    def test_disabled_and_zero_disturbances_reproduce_the_scaffold(self):
        baseline, baseline_history = propagate.nominal(ObservationSatellite())
        cases = (
            ({}, False),
            ({"obscuration_prob": 1.0, "contact_loss_prob": 1.0}, False),
            (
                {"noise_std": 0.0, "obscuration_prob": 0.0, "contact_loss_prob": 0.0},
                True,
            ),
        )
        for params, stochastic in cases:
            with self.subTest(params=params, stochastic=stochastic):
                model = DisturbedObservationSatellite(
                    p=params,
                    r={"seed": 0},
                    sp={"run_stochastic": stochastic, "track_pdf": True},
                )
                result, history = propagate.nominal(model)
                for key in baseline:
                    self.assertEqual(result[key], baseline[key])
                for key in baseline_history:
                    np.testing.assert_array_equal(history[key], baseline_history[key])
                self.assertEqual(result["tend.classify.missed"], 0)
                self.assertEqual(result["tend.classify.unprocessed"], 0)
                np.testing.assert_array_equal(history["fxns.schedule.r.probdens"], 1.0)
                self.assert_accounting(result, history)

    def test_seeded_inputs_match_numpy_draws_and_repeated_runs(self):
        histories = []
        for seed in (0, 1, 31):
            with self.subTest(seed=seed):
                model = DisturbedObservationSatellite(
                    r={"seed": seed}, sp={"run_stochastic": True}
                )
                before = copy.deepcopy(model.classify())
                state = copy.deepcopy(model.fxns["schedule"].r.rng.bit_generator.state)
                result, history = propagate.nominal(model)
                repeated, repeated_history = propagate.nominal(model)
                self.assert_same(result, repeated)
                self.assert_same(history, repeated_history)
                self.assertEqual(model.classify(), before)
                self.assertEqual(
                    model.fxns["schedule"].r.rng.bit_generator.state, state
                )
                generator = np.random.default_rng(seed)
                noise, obscured, contact_lost = np.zeros((3, 7))
                for time in (1, 2, 3, 4):
                    noise[time] = generator.normal(0.0, 0.05)
                    obscured[time] = generator.binomial(1, 0.1)
                for time in (5, 6):
                    contact_lost[time] = generator.binomial(1, 0.1)
                for name, expected in (
                    ("noise", noise),
                    ("obscured", obscured),
                    ("contact_lost", contact_lost),
                ):
                    np.testing.assert_array_equal(
                        history["fxns.schedule.r.s." + name], expected
                    )
                for time, signal in enumerate((0.2, 0.8, 0.1, 0.9), start=1):
                    expected = (
                        0.0 if obscured[time] else np.clip(signal + noise[time], 0, 1)
                    )
                    self.assertEqual(
                        history["flows.observation.s.signal"][time], expected
                    )
                histories.append(noise)
                self.assert_accounting(result, history)
        self.assertFalse(np.array_equal(histories[0], histories[1]))

    def test_probability_extremes_idle_steps_and_score_clipping(self):
        for obscured, outage, acquired, downlinked, queued in (
            (0.0, 0.0, 4, 2, 0),
            (1.0, 0.0, 0, 0, 0),
            (0.0, 1.0, 4, 0, 2),
            (1.0, 1.0, 0, 0, 0),
        ):
            with self.subTest(obscured=obscured, outage=outage):
                model = DisturbedObservationSatellite(
                    p={
                        "noise_std": 0.0,
                        "obscuration_prob": obscured,
                        "contact_loss_prob": outage,
                    },
                    sp={"run_stochastic": True, "track_pdf": True},
                )
                result, history = propagate.nominal(model)
                for name, expected in (
                    ("acquired", acquired),
                    ("downlinked", downlinked),
                    ("queued", queued),
                ):
                    self.assertEqual(result["tend.classify." + name], expected)
                np.testing.assert_array_equal(history["fxns.schedule.r.probdens"], 1.0)
                self.assert_accounting(result, history)
        empty = DisturbedObservationSatellite(
            p={"observation_times": (), "signals": (), "contact_times": ()},
            sp={"run_stochastic": True, "track_pdf": True},
        )
        before = copy.deepcopy(empty.fxns["schedule"].r.rng.bit_generator.state)
        empty(time=6.0)
        self.assertEqual(empty.fxns["schedule"].r.rng.bit_generator.state, before)
        self.assertEqual(set(empty.classify().values()), {0})
        result, history = propagate.nominal(
            DisturbedObservationSatellite(
                p={"noise_std": 10.0, "obscuration_prob": 0.0},
                r={"seed": 0},
                sp={"run_stochastic": True},
            )
        )
        scores = history["flows.observation.s.signal"][1:5]
        self.assertIn(0.0, scores)
        self.assertIn(1.0, scores)
        self.assert_accounting(result, history)

    def test_injected_faults_have_known_timing_and_counts(self):
        # function, mode, injection, acquired, selected, sent, queued, missed, unprocessed
        cases = (
            ("capture", "failed", 3, 2, 1, 1, 0, 2, 0),
            ("capture", "biased", 1, 4, 4, 2, 2, 0, 0),
            ("process", "failed", 3, 4, 1, 1, 0, 0, 2),
            ("downlink", "failed", 5, 4, 2, 0, 2, 0, 0),
            ("capture", "failed", 5, 4, 2, 2, 0, 0, 0),
            ("capture", "failed", 0, 0, 0, 0, 0, 4, 0),
        )
        for function, mode, time, *counts in cases:
            for staged in (False, True):
                with self.subTest(
                    function=function, mode=mode, time=time, staged=staged
                ):
                    model = DisturbedObservationSatellite()
                    result, history = propagate.one_fault(
                        model,
                        function,
                        mode,
                        time=time,
                        staged=staged,
                    )
                    name = next(n for n in result.nest(levels=1) if n != "nominal")
                    faulty, trace = result.get(name), history.get(name)
                    for metric, expected in zip(
                        (
                            "acquired",
                            "selected",
                            "downlinked",
                            "queued",
                            "missed",
                            "unprocessed",
                        ),
                        counts,
                    ):
                        self.assertEqual(faulty["tend.classify." + metric], expected)
                    np.testing.assert_array_equal(
                        trace["fxns." + function + ".m.faults." + mode],
                        np.arange(7) >= time,
                    )
                    self.assert_accounting(faulty, trace)

    def test_fault_samples_preserve_random_inputs_and_staged_results(self):
        model = DisturbedObservationSatellite(
            r={"seed": 0}, sp={"run_stochastic": True, "track_pdf": True}
        )
        domain = FaultDomain(model)
        for function, mode in (
            ("capture", "failed"),
            ("capture", "biased"),
            ("process", "failed"),
            ("downlink", "failed"),
        ):
            domain.add_fault(function, mode)
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([1.0, 3.0], weights=[0.25, 0.75])
        before = copy.deepcopy([s.asdict() for s in sample.scenarios()])
        reference = propagate.fault_sample(
            model, sample, staged=False, showprogress=False
        )
        staged = propagate.fault_sample(model, sample, staged=True, showprogress=False)
        for actual, expected in zip(staged, reference):
            self.assert_same(actual, expected)
        result, history = staged
        for scenario in sample.scenarios():
            trace = history.get(scenario.name)
            self.assert_accounting(result.get(scenario.name), trace)
            for field in ("noise", "obscured", "contact_lost"):
                np.testing.assert_array_equal(
                    trace["fxns.schedule.r.s." + field],
                    history["nominal.fxns.schedule.r.s." + field],
                )
        self.assertEqual([s.asdict() for s in sample.scenarios()], before)
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_joint_fault_scenario(
            (("capture", "failed"), ("capture", "biased")), 1.0
        )
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        name = sample.scenarios()[0].name
        self.assertEqual(result[name + ".tend.classify.acquired"], 0)
        self.assertEqual(result[name + ".tend.classify.missed"], 4)
        self.assert_accounting(result.get(name), history.get(name))

    def test_copy_pickle_reset_and_continuation_preserve_the_seeded_path(self):
        model = DisturbedObservationSatellite(
            r={"seed": 31}, sp={"run_stochastic": True, "track_pdf": True}
        )
        expected, expected_history = propagate.nominal(model)
        model(time=3.0)
        snapshot = model.classify()
        rng_state = copy.deepcopy(model.fxns["schedule"].r.rng.bit_generator.state)
        for clone in (model.copy(), pickle.loads(pickle.dumps(model))):
            clone(time=6.0)
            self.assertEqual(
                clone.classify(),
                {k.removeprefix("tend.classify."): v for k, v in expected.items()},
            )
            self.assert_same(clone.h.flatten(), expected_history)
            self.assertEqual(model.classify(), snapshot)
            self.assertEqual(
                model.fxns["schedule"].r.rng.bit_generator.state, rng_state
            )
        model.reset()
        model(time=6.0)
        self.assert_same(model.h.flatten(), expected_history)

    def test_input_density_tracking_does_not_change_draws(self):
        for sigma in (0.0, 0.2):
            with self.subTest(sigma=sigma):
                params = {"noise_std": sigma, "contact_times": (2, 5, 6)}
                tracked, history = propagate.nominal(
                    DisturbedObservationSatellite(
                        p=params,
                        r={"seed": 5},
                        sp={"run_stochastic": True, "track_pdf": True},
                    )
                )
                untracked, plain = propagate.nominal(
                    DisturbedObservationSatellite(
                        p=params,
                        r={"seed": 5},
                        sp={"run_stochastic": True},
                    )
                )
                self.assert_same(tracked, untracked)
                for key in history:
                    if not key.endswith("probdens"):
                        np.testing.assert_array_equal(
                            history[key], plain[key], err_msg=key
                        )
                density = []
                for time in range(7):
                    value = 1.0
                    if time in (1, 2, 3, 4):
                        x = history["fxns.schedule.r.s.noise"][time]
                        if sigma:
                            value *= math.exp(-0.5 * (x / sigma) ** 2) / (
                                sigma * math.sqrt(2 * math.pi)
                            )
                        blocked = history["fxns.schedule.r.s.obscured"][time]
                        value *= 0.1 if blocked else 0.9
                    if time in (2, 5, 6):
                        lost = history["fxns.schedule.r.s.contact_lost"][time]
                        value *= 0.1 if lost else 0.9
                    density.append(value)
                np.testing.assert_allclose(history["fxns.schedule.r.probdens"], density)
                self.assert_accounting(tracked, history)

    def test_disturbance_parameters_reject_invalid_values(self):
        for name, invalid in (
            ("noise_std", (-0.1, math.nan, math.inf)),
            ("obscuration_prob", (-0.1, 1.1, math.nan, math.inf)),
            ("contact_loss_prob", (-0.1, 1.1, math.nan, math.inf)),
            ("bias", (-1.1, 1.1, math.nan, math.inf)),
        ):
            for value in invalid:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(Exception, "outside of limits|finite"):
                        DisturbanceParam(**{name: value})


if __name__ == "__main__":
    unittest.main()

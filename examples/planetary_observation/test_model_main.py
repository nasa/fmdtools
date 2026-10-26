#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analytical checks for the deterministic planetary-observation scaffold.

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
import pickle
import unittest

import numpy as np

from fmdtools.sim import propagate
from fmdtools_examples.planetary_observation.model_main import (
    MissionParam,
    ObservationSatellite,
)


class TestSatelliteScaffold(unittest.TestCase):
    def assert_accounting(self, result, history, capacity):
        """Every selected packet is downlinked, queued or explicitly dropped."""
        selected = history["fxns.process.s.selected"]
        delivered = history["fxns.downlink.s.downlinked"]
        queued = history["flows.buffer.s.queued"]
        dropped = history["fxns.process.s.dropped"]
        np.testing.assert_array_equal(selected, delivered + queued + dropped)
        self.assertTrue(np.all((0 <= queued) & (queued <= capacity)))
        np.testing.assert_array_equal(
            history["fxns.capture.s.acquired"], history["fxns.process.s.processed"]
        )
        for field, values in (
            ("selected", selected),
            ("downlinked", delivered),
            ("queued", queued),
            ("dropped", dropped),
        ):
            self.assertEqual(result["tend.classify." + field], values[-1])

    def test_nominal_run_matches_full_hand_calculated_history(self):
        model = ObservationSatellite()
        result, history = propagate.nominal(model)
        expected = {
            "time": [0, 1, 2, 3, 4, 5, 6],
            "fxns.schedule.s.opportunities": [0, 1, 2, 3, 4, 4, 4],
            "fxns.capture.s.acquired": [0, 1, 2, 3, 4, 4, 4],
            "fxns.process.s.processed": [0, 1, 2, 3, 4, 4, 4],
            "fxns.process.s.selected": [0, 0, 1, 1, 2, 2, 2],
            "fxns.process.s.dropped": [0, 0, 0, 0, 0, 0, 0],
            "flows.buffer.s.queued": [0, 0, 1, 1, 2, 1, 0],
            "fxns.downlink.s.downlinked": [0, 0, 0, 0, 0, 1, 2],
            "flows.scene.s.visible": [False, True, True, True, True, False, False],
            "flows.scene.s.contact": [False, False, False, False, False, True, True],
            "flows.observation.s.valid": [False, True, True, True, True, False, False],
            "flows.observation.s.scene_id": [-1, 0, 1, 2, 3, -1, -1],
            "flows.observation.s.signal": [0, 0.2, 0.8, 0.1, 0.9, 0, 0],
        }
        for key, values in expected.items():
            np.testing.assert_array_equal(history[key], values, err_msg=key)
        self.assertEqual(
            dict(result),
            {
                "tend.classify.opportunities": 4,
                "tend.classify.acquired": 4,
                "tend.classify.processed": 4,
                "tend.classify.selected": 2,
                "tend.classify.downlinked": 2,
                "tend.classify.queued": 0,
                "tend.classify.dropped": 0,
            },
        )
        self.assert_accounting(result, history, 4)
        self.assertEqual(set(model.classify().values()), {0})

    def test_capacity_contact_and_selection_boundaries(self):
        # (parameter overrides, selected, downlinked, queued, dropped)
        cases = (
            ({"contact_times": ()}, 2, 0, 2, 0),
            ({"buffer_capacity": 1}, 2, 1, 0, 1),
            ({"buffer_capacity": 0}, 2, 0, 0, 2),
            ({"downlink_capacity": 0}, 2, 0, 2, 0),
            ({"contact_times": (5,), "downlink_capacity": 4}, 2, 2, 0, 0),
            ({"contact_times": (2, 4), "buffer_capacity": 1}, 2, 2, 0, 0),
            # At t=4 the buffer is still full when the second selection arrives.
            ({"contact_times": (4,), "buffer_capacity": 1}, 2, 1, 0, 1),
            ({"threshold": 0.0}, 4, 2, 2, 0),
            ({"threshold": 1.0}, 0, 0, 0, 0),
            ({"signals": (0.49, 0.5, 0.51, 0.0)}, 2, 2, 0, 0),
            ({"observation_times": (), "signals": ()}, 0, 0, 0, 0),
        )
        for params, selected, sent, queued, dropped in cases:
            with self.subTest(params=params):
                before = copy.deepcopy(params)
                model = ObservationSatellite(p=params)
                result, history = propagate.nominal(model)
                for name, count in zip(
                    ("selected", "downlinked", "queued", "dropped"),
                    (selected, sent, queued, dropped),
                ):
                    self.assertEqual(result["tend.classify." + name], count)
                self.assert_accounting(result, history, model.p.buffer_capacity)
                self.assertEqual(params, before)

    def test_irregular_schedule_clears_observations_between_opportunities(self):
        model = ObservationSatellite(
            p={
                "observation_times": (2, 5),
                "signals": (0.5, 0.9),
                "contact_times": (1, 3, 6),
            }
        )
        result, history = propagate.nominal(model)
        np.testing.assert_array_equal(
            history["flows.observation.s.signal"], [0, 0, 0.5, 0, 0, 0.9, 0]
        )
        np.testing.assert_array_equal(
            history["flows.observation.s.scene_id"], [-1, -1, 0, -1, -1, 1, -1]
        )
        np.testing.assert_array_equal(
            history["fxns.process.s.processed"], [0, 0, 1, 1, 1, 2, 2]
        )
        np.testing.assert_array_equal(
            history["fxns.downlink.s.downlinked"], [0, 0, 0, 1, 1, 1, 2]
        )
        self.assert_accounting(result, history, model.p.buffer_capacity)
        self.assertEqual(result["tend.classify.opportunities"], 2)

    def test_initial_partial_and_extended_runs_do_not_replay_packets(self):
        for end, acquired, selected, sent, queued in (
            (0, 0, 0, 0, 0),
            (3, 3, 1, 0, 1),
            (4, 4, 2, 0, 2),
            (6, 4, 2, 2, 0),
            (10, 4, 2, 2, 0),
        ):
            with self.subTest(end=end):
                model = ObservationSatellite(sp={"end_time": float(end)})
                result, history = propagate.nominal(model)
                for name, value in zip(
                    ("acquired", "selected", "downlinked", "queued"),
                    (acquired, selected, sent, queued),
                ):
                    self.assertEqual(result["tend.classify." + name], value)
                self.assert_accounting(result, history, model.p.buffer_capacity)
                np.testing.assert_array_equal(history["time"], np.arange(end + 1))

    def test_flows_are_shared_inside_the_model_and_isolated_between_copies(self):
        model = ObservationSatellite()
        self.assertEqual(
            list(model.dynamicsims), ["schedule", "capture", "process", "downlink"]
        )
        self.assertFalse(model.staticsims)
        for name, flow, functions in (
            ("scene", model.flows["scene"], ("schedule", "capture", "downlink")),
            ("observation", model.flows["observation"], ("capture", "process")),
            ("buffer", model.flows["buffer"], ("process", "downlink")),
        ):
            for function in functions:
                self.assertIs(getattr(model.fxns[function], name), flow)
        model(time=3.0)
        snapshot = model.classify()
        for clone in (model.copy(), pickle.loads(pickle.dumps(model))):
            self.assertEqual(clone.classify(), snapshot)
            self.assertIs(clone.fxns["process"].buffer, clone.flows["buffer"])
            self.assertIsNot(clone.flows["buffer"], model.flows["buffer"])
            clone(time=6.0)
            self.assertEqual(clone.classify()["downlinked"], 2)
            self.assertEqual(model.classify(), snapshot)
        # Re-entering the same time must not repeat any dynamic counter updates.
        model(time=3.0)
        self.assertEqual(model.classify(), snapshot)
        model.reset()
        model(time=6.0)
        fresh = ObservationSatellite()
        fresh(time=6.0)
        self.assertEqual(model.classify(), fresh.classify())

    def test_repeated_nominal_runs_leave_the_source_unchanged(self):
        model = ObservationSatellite()
        before = model.classify(), model.p.asdict()
        first, first_history = propagate.nominal(model)
        second, second_history = propagate.nominal(model)
        self.assertEqual(first, second)
        self.assertEqual(set(first_history), set(second_history))
        for key in first_history:
            np.testing.assert_equal(first_history[key], second_history[key])
        self.assertEqual((model.classify(), model.p.asdict()), before)
        self.assertEqual(MissionParam().copy(), MissionParam())

    def test_invalid_schedules_values_and_capacities_fail_before_simulation(self):
        cases = (
            {"observation_times": (1, 1, 3, 4)},
            {"observation_times": (0, 2, 3, 4)},
            {"observation_times": (1, 2, 4, 3)},
            {"observation_times": (1, 2.5, 3, 4)},
            {"contact_times": (-1,)},
            {"contact_times": (True,)},
            {"signals": (0.5,)},
            {"signals": (0.1, 0.2, float("nan"), 0.4)},
            {"signals": (0.1, 0.2, 0.3, 1.1)},
            {"buffer_capacity": 1.5},
            {"buffer_capacity": np.float64(1.5)},
            {"downlink_capacity": True},
            {"downlink_capacity": np.bool_(True)},
        )
        for params in cases:
            with self.subTest(params=params), self.assertRaises(ValueError):
                MissionParam(**params)
        # Scalar bounds use Parameter's declarative limits and its own error.
        for name, values in (
            ("threshold", (-0.1, 1.1, float("nan"), float("inf"), -float("inf"))),
            ("buffer_capacity", (-1,)),
            ("downlink_capacity", (-1,)),
        ):
            for value in values:
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(Exception, "outside of limits"):
                        MissionParam(**{name: value})
        self.assertEqual(MissionParam(buffer_capacity=np.int64(2)).buffer_capacity, 2)
        for options in (
            {"sp": {"start_time": 1.0}},
            {"sp": {"units": "sec"}},
            {"sp": {"dt": 2.0}},
            {"t": {"dt": 0.5}},
            {"sp": {"end_time": 2.5}},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ObservationSatellite(**options)


if __name__ == "__main__":
    unittest.main()

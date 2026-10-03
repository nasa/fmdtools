#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for confidence intervals from unequal-length histories.

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

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bootstrap

from fmdtools.analyze.history import History
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.propagate import Simulation


OPTIONS = {"n_resamples": 199, "random_state": 23}


def make_history(time="time", shared_time=False, as_lists=False):
    values = [
        np.arange(length, dtype=float) * scale + offset
        for length, scale, offset in ((6, 1.0, 1.0), (4, 2.0, 3.0), (5, 3.0, 5.0))
    ]
    result = {f"run{i}.value": data for i, data in enumerate(values)}
    if shared_time:
        result[time] = np.arange(6, dtype=float)
    else:
        result.update(
            {
                f"run{i}." + time: np.arange(len(data), dtype=float)
                for i, data in enumerate(values)
            }
        )
    if as_lists:
        result = {k: v.tolist() for k, v in result.items()}
    return History(result), values


class TestRaggedConfidenceHistories(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def assert_interval(self, actual, data, times, method=np.mean, arguments=None):
        arguments = arguments or {}

        def statistic(values, axis=0):
            return method(values, axis=axis, **arguments)

        algorithm = "basic" if np.any(np.all(data == data[0], axis=0)) else "BCa"
        expected = bootstrap(
            (data,),
            statistic,
            axis=0,
            method=algorithm,
            confidence_level=0.8,
            **OPTIONS,
        )
        np.testing.assert_allclose(actual["stat"], statistic(data))
        np.testing.assert_allclose(actual["low"], expected.confidence_interval.low)
        np.testing.assert_allclose(actual["high"], expected.confidence_interval.high)
        np.testing.assert_array_equal(actual[times[0]], times[1])
        self.assertEqual(len(actual["stat"]), len(times[1]))
        self.assertIs(type(actual), History)

    def test_clip_each_trace_before_stacking_and_averaging_times(self):
        for cutoff in ("max", None, 1, 3, 4, 10, -1):
            for as_lists in (False, True):
                for shared_time in (False, True):
                    with self.subTest(
                        cutoff=cutoff, lists=as_lists, shared=shared_time
                    ):
                        history, values = make_history(
                            as_lists=as_lists, shared_time=shared_time
                        )
                        before = history.copy()
                        end = (
                            4 if cutoff == "max" else slice(None, cutoff).indices(4)[1]
                        )
                        actual = history.get_mean_ci_errhist(
                            "value", ci=0.8, max_ind=cutoff, **OPTIONS
                        )
                        data = np.array([v[:end] for v in values])
                        self.assert_interval(actual, data, ("time", np.arange(end)))
                        self.assertEqual(history, before)

    def test_nested_histories_and_custom_time_keys_preserve_averaging(self):
        history, values = make_history(time="elapsed")
        history["run2.elapsed"] += 3.0
        history["diagnostic"] = np.arange(3)
        for nested in (False, True):
            with self.subTest(nested=nested):
                output = history.nest() if nested else history
                before = output.copy()
                actual = output.get_mean_ci_errhist(
                    "value", time="elapsed", ci=0.8, **OPTIONS
                )
                self.assert_interval(
                    actual,
                    np.array([v[:3] for v in values]),
                    ("elapsed", np.arange(3) + 1.0),
                )
                self.assertEqual(output, before)

    def test_shorter_time_vectors_limit_the_shared_available_prefix(self):
        history, values = make_history(shared_time=True)
        history["time"] = np.array([0.0, 1.0, 2.0])
        for cutoff in ("max", 10, -1):
            with self.subTest(cutoff=cutoff):
                end = 2 if cutoff == -1 else 3
                actual = history.get_mean_ci_errhist(
                    "value", max_ind=cutoff, ci=0.8, **OPTIONS
                )
                self.assert_interval(
                    actual,
                    np.array([v[:end] for v in values]),
                    ("time", np.arange(end)),
                )

    def test_configured_statistic_and_resampling_controls_survive_clipping(self):
        history, values = make_history()
        actual = history.get_mean_ci_errhist(
            "value", ci=0.8, max_ind=3, method=np.quantile, q=0.75, **OPTIONS
        )
        self.assert_interval(
            actual,
            np.array([v[:3] for v in values]),
            ("time", np.arange(3)),
            method=np.quantile,
            arguments={"q": 0.75},
        )

    def test_real_unequal_duration_simulations_and_public_plot(self):
        histories = {}
        for i, (end_time, scale) in enumerate(((2.0, 1.0), (3.0, 2.0), (4.0, 4.0))):
            model = ExampleFunction(sp={"end_time": end_time}, p={"x": scale})
            _, histories[f"run{i}"] = Simulation(mdl=model)()
        combined = History(histories).flatten()
        before = combined.copy()
        self.assertEqual([len(h.time) for h in histories.values()], [3, 4, 5])
        data = np.array([h["s.x"][:3] for h in histories.values()])
        actual = combined.get_mean_ci_errhist("s.x", ci=0.8, **OPTIONS)
        self.assert_interval(actual, data, ("time", np.arange(3)))
        fig, ax = combined.plot_mean_ci_line("s.x")
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(3))
        np.testing.assert_allclose(ax.lines[0].get_ydata(), data.mean(axis=0))
        fig.canvas.draw()
        self.assertEqual(combined, before)


if __name__ == "__main__":
    unittest.main()

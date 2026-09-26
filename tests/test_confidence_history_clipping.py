#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for time clipping in confidence-interval histories.

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

from fmdtools.analyze.common import calc_metric_ci, plot_err_hist
from fmdtools.analyze.history import History


class TestConfidenceHistoryTimeClipping(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def make_history(self, time="time", repeated_times=False, as_lists=False):
        values = np.array(
            [
                [0.0, 10.0, 20.0, 30.0, 40.0],
                [2.0, 12.0, 22.0, 32.0, 42.0],
                [1.0, 11.0, 21.0, 31.0, 41.0],
            ]
        )
        times = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
        result = {f"run{i}.signal": row.copy() for i, row in enumerate(values)}
        if repeated_times:
            result.update({f"run{i}." + time: times.copy() for i in range(3)})
        else:
            result[time] = times.copy()
        if as_lists:
            result = {key: value.tolist() for key, value in result.items()}
        return History(result), values, times

    def test_explicit_cutoff_aligns_times_statistics_and_bounds(self):
        for cutoff in (1, 2, 4, 5, 10, -1):
            for as_lists in (False, True):
                with self.subTest(cutoff=cutoff, as_lists=as_lists):
                    history, values, times = self.make_history(as_lists=as_lists)
                    before = history.copy()
                    result = history.get_mean_ci_errhist(
                        "signal",
                        ci=0.8,
                        max_ind=cutoff,
                        n_resamples=199,
                        rng=np.random.default_rng(12),
                    )
                    expected = calc_metric_ci(
                        values[:, :cutoff],
                        confidence_level=0.8,
                        axis=0,
                        n_resamples=199,
                        rng=np.random.default_rng(12),
                    )
                    self.assertIs(type(result), History)
                    np.testing.assert_array_equal(result.time, times[:cutoff])
                    for key, value in zip(("stat", "low", "high"), expected):
                        np.testing.assert_array_equal(result[key], value)
                        self.assertEqual(result[key].shape, result.time.shape)
                    for key in history:
                        np.testing.assert_array_equal(history[key], before[key])

    def test_custom_and_replicate_time_vectors_are_clipped_after_aggregation(self):
        for repeated_times in (False, True):
            with self.subTest(repeated_times=repeated_times):
                history, _, times = self.make_history(
                    time="elapsed", repeated_times=repeated_times
                )
                if repeated_times:
                    history["run2.elapsed"] += 3.0
                    times = times + 1.0
                result = history.get_mean_ci_errhist(
                    "signal",
                    max_ind=3,
                    time="elapsed",
                    n_resamples=99,
                    rng=np.random.default_rng(2),
                )
                self.assertEqual(set(result), {"elapsed", "stat", "low", "high"})
                np.testing.assert_array_equal(result.elapsed, times[:3])
                np.testing.assert_array_equal(result.stat, [1.0, 11.0, 21.0])

    def test_automatic_shorter_cutoff_also_clips_the_time_vector(self):
        history, _, times = self.make_history()
        history["diagnostic"] = [7.0, 8.0, 9.0]
        result = history.get_mean_ci_errhist(
            "signal", n_resamples=99, rng=np.random.default_rng(4)
        )
        np.testing.assert_array_equal(result.time, times[:3])
        for key in ("stat", "low", "high"):
            self.assertEqual(len(result[key]), 3)

    def test_unclipped_default_matches_explicit_full_length(self):
        history, _, times = self.make_history()
        default = history.get_mean_ci_errhist(
            "signal", n_resamples=99, rng=np.random.default_rng(3)
        )
        full = history.get_mean_ci_errhist(
            "signal", max_ind=len(times), n_resamples=99, rng=np.random.default_rng(3)
        )
        for key in default:
            np.testing.assert_array_equal(default[key], full[key])
        np.testing.assert_array_equal(default.time, times)

    def test_public_plot_and_direct_error_plot_render_clipped_histories(self):
        for time in ("time", "elapsed"):
            for boundtype in ("fill", "line"):
                for public_method in (False, True):
                    with self.subTest(
                        time=time, boundtype=boundtype, public_method=public_method
                    ):
                        history, values, times = self.make_history(time=time)
                        fig, ax = plt.subplots()
                        if public_method:
                            out_fig, out_ax = history.plot_mean_ci_line(
                                "signal",
                                max_ind=3,
                                time=time,
                                fig=fig,
                                ax=ax,
                                boundtype=boundtype,
                                label="mean",
                            )
                        else:
                            result = history.get_mean_ci_errhist(
                                "signal",
                                max_ind=3,
                                time=time,
                                n_resamples=99,
                                rng=np.random.default_rng(3),
                            )
                            out_fig, out_ax = plot_err_hist(
                                result,
                                time=time,
                                fig=fig,
                                ax=ax,
                                boundtype=boundtype,
                                label="mean",
                            )
                        self.assertIs(out_fig, fig)
                        self.assertIs(out_ax, ax)
                        np.testing.assert_array_equal(
                            ax.lines[0].get_xdata(), times[:3]
                        )
                        np.testing.assert_array_equal(
                            ax.lines[0].get_ydata(), values.mean(axis=0)[:3]
                        )
                        np.testing.assert_allclose(ax.get_xlim(), (times[0], times[2]))
                        self.assertEqual(ax.lines[0].get_label(), "mean")
                        if boundtype == "line":
                            for line in ax.lines[1:]:
                                np.testing.assert_array_equal(
                                    line.get_xdata(), times[:3]
                                )
                        else:
                            vertices = ax.collections[0].get_paths()[0].vertices
                            np.testing.assert_array_equal(
                                np.unique(vertices[:, 0]), times[:3]
                            )
                        fig.canvas.draw()
                        plt.close(fig)


if __name__ == "__main__":
    unittest.main()

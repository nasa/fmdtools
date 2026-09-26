#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for error-history plot axes and time coordinates.

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

from fmdtools.analyze.common import plot_err_hist, plot_err_lines
from fmdtools.analyze.history import History


class TestErrorHistoryPlots(unittest.TestCase):
    """Check plotted data and real canvas rendering without image snapshots."""

    def tearDown(self):
        plt.close("all")

    def make_error_history(self, times=(10.0, 20.0, 30.0), time_key="time"):
        return History(
            {
                time_key: np.asarray(times),
                "stat": np.array([1.0, 2.0, 3.0]),
                "low": np.array([0.0, 1.0, 2.0]),
                "high": np.array([2.0, 3.0, 4.0]),
            }
        )

    def assert_coordinates(self, history, ax, time_key="time", medians=False):
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), history[time_key])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), history.stat)
        expected_limits = (history[time_key][0], history[time_key][-1])
        np.testing.assert_allclose(ax.get_xlim(), expected_limits)
        if ax.collections:
            self.assertEqual(len(ax.collections), 2 if medians else 1)
            for collection in ax.collections:
                x_coords = collection.get_paths()[0].vertices[:, 0]
                np.testing.assert_array_equal(np.unique(x_coords), history[time_key])
        else:
            bounds = [history.high, history.low]
            if medians:
                bounds.extend([history.med_high, history.med_low])
            self.assertEqual(len(ax.lines), len(bounds) + 1)
            for line, values in zip(ax.lines[1:], bounds):
                np.testing.assert_array_equal(line.get_xdata(), history[time_key])
                np.testing.assert_array_equal(line.get_ydata(), values)
        ax.figure.canvas.draw()

    def test_new_plots_are_two_dimensional_and_use_requested_size(self):
        for boundtype in ("fill", "line"):
            with self.subTest(boundtype=boundtype):
                history = self.make_error_history()
                fig, ax = plot_err_hist(history, boundtype=boundtype, figsize=(8, 3))
                self.assertEqual(ax.name, "rectilinear")
                self.assertIs(ax.figure, fig)
                np.testing.assert_allclose(fig.get_size_inches(), [8, 3])
                self.assert_coordinates(history, ax)

    def test_existing_axes_use_times_for_both_centre_and_bounds(self):
        for times in ((10.0, 20.0, 30.0), (0.0, 0.25, 0.5), (-2.0, -0.5, 1.0)):
            for boundtype in ("fill", "line"):
                with self.subTest(times=times, boundtype=boundtype):
                    history = self.make_error_history(times)
                    original = history.copy()
                    fig, ax = plt.subplots()
                    result_fig, result_ax = plot_err_hist(
                        history,
                        fig=fig,
                        ax=ax,
                        boundtype=boundtype,
                        label="measurement",
                        linestyle=":",
                        marker="o",
                        linewidth=2,
                    )
                    self.assertIs(result_fig, fig)
                    self.assertIs(result_ax, ax)
                    self.assertEqual(len(fig.axes), 1)
                    self.assertEqual(ax.lines[0].get_label(), "measurement")
                    self.assertEqual(ax.lines[0].get_linestyle(), ":")
                    self.assertEqual(ax.lines[0].get_marker(), "o")
                    self.assert_coordinates(history, ax)
                    for key in history:
                        np.testing.assert_array_equal(history[key], original[key])

    def test_custom_time_and_inner_bounds_use_the_same_coordinates(self):
        for boundtype in ("fill", "line"):
            with self.subTest(boundtype=boundtype):
                history = self.make_error_history(time_key="elapsed")
                history["med_low"] = history.stat - 0.5
                history["med_high"] = history.stat + 0.5
                fig, ax = plt.subplots()
                plot_err_hist(
                    history, fig=fig, ax=ax, time="elapsed", boundtype=boundtype
                )
                self.assert_coordinates(history, ax, time_key="elapsed", medians=True)

    def test_existing_figure_is_preserved_when_creating_axes(self):
        history = self.make_error_history()
        fig = plt.figure(figsize=(7, 2))
        result_fig, ax = plot_err_hist(history, fig=fig)
        self.assertIs(result_fig, fig)
        self.assertEqual(ax.name, "rectilinear")
        self.assertEqual(len(fig.axes), 1)
        self.assert_coordinates(history, ax)

    def test_error_lines_work_with_new_and_existing_axes(self):
        history = self.make_error_history()
        for existing in (False, True):
            with self.subTest(existing=existing):
                fig, ax = plt.subplots() if existing else (None, None)
                out_fig, out_ax = plot_err_lines(
                    history.time,
                    history.low,
                    history.high,
                    fig=fig,
                    ax=ax,
                    figsize=(8, 3),
                )
                if existing:
                    self.assertIs(out_fig, fig)
                    self.assertIs(out_ax, ax)
                else:
                    np.testing.assert_allclose(out_fig.get_size_inches(), [8, 3])
                self.assertEqual(out_ax.name, "rectilinear")
                self.assertEqual(len(out_ax.lines), 2)
                for line, values in zip(out_ax.lines, (history.high, history.low)):
                    np.testing.assert_array_equal(line.get_xdata(), history.time)
                    np.testing.assert_array_equal(line.get_ydata(), values)
                out_fig.canvas.draw()

    def test_public_history_plot_methods_use_actual_timestamps(self):
        history = History(
            {
                "first.signal": [1.0, 3.0, 5.0],
                "second.signal": [3.0, 5.0, 7.0],
                "third.signal": [2.0, 4.0, 6.0],
                "time": [10.0, 12.0, 18.0],
            }
        )
        for method in (
            "plot_mean_std_line",
            "plot_mean_bound_line",
            "plot_percentile_line",
            "plot_mean_ci_line",
        ):
            for boundtype in ("fill", "line"):
                with self.subTest(method=method, boundtype=boundtype):
                    fig, ax = getattr(history, method)("signal", boundtype=boundtype)
                    self.assertEqual(ax.name, "rectilinear")
                    np.testing.assert_array_equal(ax.lines[0].get_xdata(), history.time)
                    np.testing.assert_array_equal(
                        ax.lines[0].get_ydata(), [2.0, 4.0, 6.0]
                    )
                    fig.canvas.draw()

    def test_unit_spaced_times_keep_existing_data_and_plot_options(self):
        history = self.make_error_history(times=(0.0, 1.0, 2.0))
        fig, ax = plt.subplots()
        plot_err_hist(
            history,
            fig=fig,
            ax=ax,
            xlabel="Elapsed",
            ylabel="Signal",
            title="Example",
            xlim=(-1, 3),
            ylim=(-2, 5),
        )
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), history.stat)
        self.assertEqual(ax.get_xlabel(), "Elapsed")
        self.assertEqual(ax.get_ylabel(), "Signal")
        self.assertEqual(ax.get_title(), "Example")
        np.testing.assert_allclose(ax.get_xlim(), [-1, 3])
        np.testing.assert_allclose(ax.get_ylim(), [-2, 5])
        fig.canvas.draw()


if __name__ == "__main__":
    unittest.main()

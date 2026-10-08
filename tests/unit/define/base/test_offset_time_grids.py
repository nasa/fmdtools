#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for retaining a time grid's nonzero origin.

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

import copy
import itertools
import unittest

import numpy as np

from fmdtools.analyze.phases import PhaseMap, find_interval_overlap, join_phasemaps
from fmdtools.define.base import gen_timerange
from fmdtools.define.block.base import SimParam
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestOffsetTimeGrids(unittest.TestCase):
    def test_grid_origin_and_spacing_are_preserved(self):
        for start, dt, n in itertools.product(
            (0.1, 0.25, 1.5, -1.5), (0.2, 0.5, 1.0), (0, 1, 4)
        ):
            with self.subTest(start=start, dt=dt, n=n):
                end = start + n * dt
                expected = np.round(start + np.arange(n + 1) * dt, 7)
                actual = gen_timerange(start, end, dt)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual[0], round(start, 7))
                self.assertEqual(len(np.unique(actual)), len(actual))
                if n:
                    np.testing.assert_allclose(np.diff(actual), dt)

    def test_unaligned_end_never_shifts_an_included_sample(self):
        for start, end, dt in (
            (0.25, 1.1, 0.5),
            (0.1, 0.65, 0.2),
            (1.5, 3.25, 1.0),
            (-0.25, 0.4, 0.5),
        ):
            with self.subTest(start=start, end=end, dt=dt):
                expected = []
                i = 0
                while start + i * dt <= end:
                    expected.append(round(start + i * dt, 7))
                    i += 1
                actual = gen_timerange(start, end, dt)
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.all((actual >= start) & (actual <= end)))

    def test_zero_origin_integer_grids_empty_intervals_and_rounding_controls(self):
        for start, end, dt in ((0.0, 1.0, 0.1), (0.0, 4.0, 1.0), (2.0, 4.0, 0.5)):
            expected = np.round(np.arange(start, end + 1e-7, dt), 7)
            np.testing.assert_array_equal(gen_timerange(start, end, dt), expected)
        self.assertEqual(gen_timerange(2.0, 1.0, 0.5).size, 0)
        np.testing.assert_array_equal(
            gen_timerange(0.123, 0.523, 0.2, min_r=3), [0.123, 0.323, 0.523]
        )
        params = SimParam(start_time=0.25, end_time=1.25, dt=0.5)
        np.testing.assert_array_equal(params.get_timerange(), [0.25, 0.75, 1.25])
        np.testing.assert_array_equal(
            params.get_timerange(start_time=0.0, end_time=1.0), [0.0, 0.5, 1.0]
        )

    def test_phase_intersections_keep_offsets_and_reject_disjoint_lattices(self):
        first = PhaseMap({"run": [0.25, 1.75]}, dt=0.5)
        second = PhaseMap({"on": [0.75, 2.25]}, dt=0.5)
        before = copy.deepcopy([first.phases, second.phases])
        for a, b in ((first, second), (second, first)):
            joined = join_phasemaps(a, b)
            key = next(iter(joined.phases))
            np.testing.assert_array_equal(
                joined.get_phase_times(key), [0.75, 1.25, 1.75]
            )
            self.assertEqual(joined.calc_phase_time(key), 1.5)
        self.assertEqual(find_interval_overlap([0.0, 3.0], [0.5, 2.5], dt=1.0), [])
        self.assertEqual([first.phases, second.phases], before)

    def test_fault_sample_times_stay_inside_the_declared_offset_phase(self):
        model = ExampleFunction(sp={"start_time": 0.25, "end_time": 1.75, "dt": 0.5})
        domain = FaultDomain(model)
        domain.add_fault(model.name, "low")
        phases = PhaseMap({"run": [0.25, 1.75]}, dt=0.5)
        for method, args in (
            ("all", ()),
            ("even", (1,)),
            ("quad", ([-0.5, 0.5], [1.0, 1.0])),
        ):
            sample = FaultSample(domain, phasemap=phases, def_mdl_phasemap=False)
            sample.add_fault_phases("run", method=method, args=args)
            times = [s.time for s in sample.scenarios()]
            self.assertTrue(times)
            self.assertTrue(all(0.25 <= t <= 1.75 for t in times))
            for t in times:
                self.assertAlmostEqual((t - 0.25) / 0.5, round((t - 0.25) / 0.5))
            self.assertEqual(len(times), len(set(times)))
            if method == "all":
                np.testing.assert_array_equal(times, [0.25, 0.75, 1.25, 1.75])
            self.assertAlmostEqual(
                sum(s.rate for s in sample.scenarios()), model.m.get_fault("low").prob
            )


if __name__ == "__main__":
    unittest.main()

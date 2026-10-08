#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for numerical correctness.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
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

from fmdtools.analyze.history import History
from fmdtools.define.block.function import Function
from fmdtools.define.container.parameter import Parameter
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class MixedState(State):
    scalar: np.float64 = 0.0
    matrix: np.array = np.zeros((2, 2))


class MixedParameters(Parameter, readonly=True):
    faulty: bool = False


class MixedFunction(Function):
    container_p = MixedParameters
    container_s = MixedState

    def dynamic_behavior(self):
        self.s.scalar = float(self.p.faulty and self.t.time == 3.0)
        self.s.matrix[:] = 0.0
        if self.p.faulty and self.t.time == 2.0:
            self.s.matrix[0, 1] = 1.0

    def classify(self, **kwargs):
        return {"value": self.s.scalar + self.s.matrix.sum()}


class TestMixedShapeDegradation(unittest.TestCase):
    def test_mixed_scalar_and_array_fields_preserve_every_timestamp(self):
        for shape in ((2,), (2, 3), (1, 2, 2)):
            for ntimes in (1, 4, 5):
                with self.subTest(shape=shape, ntimes=ntimes):
                    scalar = np.zeros(ntimes)
                    array = np.zeros((ntimes, *shape))
                    changed = array.copy()
                    changed[ntimes // 2].flat[-1] = 1.0
                    nominal = History(
                        {
                            "time": np.arange(ntimes),
                            "s.scalar": scalar.copy(),
                            "s.array": array.copy(),
                        }
                    )
                    faulty = History(
                        {
                            "time": np.arange(ntimes),
                            "s.scalar": scalar.copy(),
                            "s.array": changed.copy(),
                        }
                    )
                    expected = np.zeros(ntimes, dtype=bool)
                    expected[ntimes // 2] = True
                    result = faulty.get_degraded_hist("s", nomhist=nominal)
                    self.assertEqual(result["s"].shape, (ntimes,))
                    np.testing.assert_array_equal(result["s"], expected)
                    np.testing.assert_array_equal(result["total"], expected.astype(int))
                    np.testing.assert_array_equal(result["time"], np.arange(ntimes))
                    np.testing.assert_array_equal(nominal["s.array"], array)
                    np.testing.assert_array_equal(faulty["s.array"], changed)

    def test_tolerances_and_multiple_groups_keep_temporal_locality(self):
        nominal = History(
            {
                "time": np.arange(4),
                "s.scalar": np.zeros(4),
                "s.array": np.zeros((4, 2)),
                "other.scalar": np.zeros(4),
            }
        )
        faulty = nominal.copy()
        faulty["s.array"][1, 0] = 0.25
        faulty["s.scalar"][2] = 2.0
        faulty["other.scalar"][2:] = 1.0
        result = faulty.get_degraded_hist("s", "other", nomhist=nominal, difftype=0.5)
        np.testing.assert_array_equal(result["s"], [False, False, True, False])
        np.testing.assert_array_equal(result["other"], [False, False, True, True])
        np.testing.assert_array_equal(result["total"], [0, 0, 2, 1])
        reduced = faulty.get_degraded_hist(
            "s", nomhist=nominal, withtime=False, withtotal=False
        )
        self.assertEqual(list(reduced), ["s"])
        np.testing.assert_array_equal(reduced["s"], [False, True, True, False])

    def test_numeric_operators_reduce_each_field_at_each_time(self):
        scalar = np.array([0.0, 2.0, 0.0, 4.0])
        vector = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 3.0], [2.0, 2.0]])
        nominal = History(
            {"time": np.arange(4), "s.scalar": scalar, "s.vector": vector}
        )
        faulty = History(
            {
                "time": np.arange(4),
                "s.scalar": np.zeros(4),
                "s.vector": np.zeros((4, 2)),
            }
        )
        for operator, expected in (
            (np.sum, [0.0, 3.0, 6.0, 8.0]),
            (np.mean, [0.0, 1.25, 1.5, 3.0]),
            (np.max, [0.0, 2.0, 3.0, 4.0]),
            (np.all, [False, False, False, True]),
        ):
            with self.subTest(operator=operator.__name__):
                result = faulty.get_degraded_hist(
                    "s", nomhist=nominal, operator=operator, difftype="diff"
                )
                np.testing.assert_array_equal(result["s"], expected)
                self.assertEqual(result["s"].shape, (4,))

    def test_scalar_and_equal_shape_array_results_remain_unchanged(self):
        for shape in ((), (2,), (2, 2)):
            nominal = History(
                {
                    "time": np.arange(4),
                    "s.a": np.zeros((4, *shape)),
                    "s.b": np.zeros((4, *shape)),
                }
            )
            faulty = nominal.copy()
            faulty["s.a"][1] = 1.0
            with self.subTest(shape=shape):
                result = faulty.get_degraded_hist("s", nomhist=nominal)
                expected = np.zeros((4, *shape), dtype=bool)
                expected[1] = True
                np.testing.assert_array_equal(result["s"], expected)

    def test_different_start_times_are_aligned_before_shape_reduction(self):
        nominal = History(
            {"time": np.arange(5), "s.scalar": np.zeros(5), "s.array": np.zeros((5, 2))}
        )
        faulty = History(
            {
                "time": np.arange(2, 5),
                "s.scalar": np.zeros(3),
                "s.array": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
            }
        )
        result = faulty.get_degraded_hist("s", nomhist=nominal)
        np.testing.assert_array_equal(result["time"], [2, 3, 4])
        np.testing.assert_array_equal(result["s"], [False, True, False])

    def test_real_simulation_mixed_states_keep_fault_and_recovery_times(self):
        _, nominal = Simulation(mdl=MixedFunction(sp={"end_time": 4.0}))()
        _, faulty = Simulation(
            mdl=MixedFunction(sp={"end_time": 4.0}, p={"faulty": True})
        )()
        result = faulty.get_degraded_hist("s", nomhist=nominal)
        np.testing.assert_array_equal(result["time"], [0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result["s"], [False, False, True, True, False])
        np.testing.assert_array_equal(result["total"], [0, 0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()

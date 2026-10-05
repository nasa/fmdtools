#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for joined phase-map timesteps.

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

from fmdtools.analyze.phases import PhaseMap, join_phasemaps
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.define.container.mode import Fault
from fmdtools.sim.sample import FaultDomain, JointFaultSample


class TestJoinedPhaseTimestep(unittest.TestCase):
    """Joined phases must retain their source grid and exposure durations."""

    def test_intersections_match_a_discrete_lattice(self):
        for dt in (0.1, 0.25, 0.5, 1.0, 2.0):
            with self.subTest(dt=dt):
                maps = [
                    PhaseMap({"run": [0, 4 * dt], "stop": [5 * dt, 8 * dt]}, dt=dt),
                    PhaseMap({"run": [dt, 6 * dt]}, dt=dt),
                ]
                original = copy.deepcopy([p.phases for p in maps])
                for order in itertools.permutations(maps):
                    joined = join_phasemaps(*order)
                    self.assertEqual(joined.dt, dt)
                    for names in itertools.product(*(p.phases for p in order)):
                        lattice = []
                        for p, name in zip(order, names):
                            start, stop = p.phases[name]
                            lattice.append(
                                set(range(round(start / dt), round(stop / dt) + 1))
                            )
                        common = set.intersection(*lattice)
                        if not common:
                            self.assertNotIn(names, joined.phases)
                            continue
                        expected = np.array(sorted(common)) * dt
                        np.testing.assert_allclose(
                            joined.get_phase_times(names), expected
                        )
                        self.assertAlmostEqual(
                            joined.calc_phase_time(names), len(common) * dt
                        )
                        rate = Fault(prob=0.2, units="hr").calc_rate(
                            expected[0], phasemap=joined, sim_time=20, sim_units="hr"
                        )
                        self.assertAlmostEqual(rate, 0.2 * len(common) * dt)
                self.assertEqual([p.phases for p in maps], original)

    def test_single_and_empty_maps_preserve_timestep(self):
        for maps in (
            [PhaseMap({"on": [0.5, 1.0]}, dt=0.5)],
            [PhaseMap({}, dt=0.5), PhaseMap({"on": [0, 1]}, dt=0.5)],
            [PhaseMap({"on": [0, 1]}, dt=0.5), PhaseMap({"off": [2, 3]}, dt=0.5)],
        ):
            with self.subTest(maps=maps):
                joined = join_phasemaps(*maps)
                self.assertEqual(joined.dt, 0.5)
                self.assertEqual(joined.modephases, {})
                if len(maps) == 1:
                    np.testing.assert_allclose(
                        joined.get_phase_times(("on",)), [0.5, 1.0]
                    )
                else:
                    self.assertEqual(joined.phases, {})
        self.assertEqual(join_phasemaps().phases, {})
        self.assertEqual(join_phasemaps().dt, 1.0)

    def test_mismatched_timesteps_are_rejected(self):
        maps = [PhaseMap({"on": [0, 1]}, dt=0.5), PhaseMap({"on": [0, 1]}, dt=1.0)]
        for order in itertools.permutations(maps):
            with self.subTest(order=order):
                with self.assertRaisesRegex(ValueError, "timestep"):
                    join_phasemaps(*order)

    def test_joint_sample_uses_fractional_times_and_rates(self):
        model = ExFxnArch(sp={"end_time": 2.0, "dt": 0.5})
        first, second = FaultDomain(model), FaultDomain(model)
        first.add_fault("ex_fxn", "low")
        second.add_fault("ex_fxn2", "no_charge")
        sample = JointFaultSample(
            first,
            second,
            phasemaps=[
                PhaseMap({"first": [0, 1]}, dt=0.5),
                PhaseMap({"second": [0.5, 1.5]}, dt=0.5),
            ],
        )
        sample.add_fault_phases(method="all", n_joint=2, baserate="max")
        self.assertEqual(sorted(sample.get_times()), [0.5, 1.0])
        self.assertEqual(
            [s.phase for s in sample.scenarios()], [("first", "second")] * 2
        )
        self.assertAlmostEqual(sum(s.rate for s in sample.scenarios()), 1.0)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for sampling and optimization inputs.

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
from types import MappingProxyType

import numpy as np

from fmdtools.analyze.phases import PhaseMap
from fmdtools.analyze.tabulate import FMEA
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultSample, JointFaultSample, SampleApproach
from tests.test_default_fault_sample_timestep import make_domain


def make_approach(dt):
    domain = make_domain(dt)
    approach = SampleApproach(domain.mdl)
    approach.add_faultdomain("both", "faults", ("risk", "first"), ("risk", "second"))
    approach.add_faultdomain("left", "fault", "risk", "first")
    approach.add_faultdomain("right", "fault", "risk", "second")
    return approach


class TestApproachModelTimestep(unittest.TestCase):
    def test_model_registry_uses_the_simulation_grid_and_phase_exposure(self):
        for dt in (0.25, 0.5, 1.0, 2.0):
            with self.subTest(dt=dt):
                approach = make_approach(dt)
                mapping = approach.phasemaps["mdl"]
                self.assertEqual(mapping.dt, dt)
                self.assertEqual(
                    mapping.phases, {"early": [0.0, dt], "late": [2 * dt, 4 * dt]}
                )
                np.testing.assert_array_equal(
                    mapping.get_phase_times("early"), np.arange(2) * dt
                )
                np.testing.assert_array_equal(
                    mapping.get_phase_times("late"), np.arange(2, 5) * dt
                )
                self.assertEqual(mapping.calc_phase_time("early"), 2 * dt)
                self.assertEqual(mapping.calc_phase_time("late"), 3 * dt)

    def test_registered_map_matches_explicit_sampling_for_every_domain_form(self):
        for dt in (0.25, 0.5, 1.0, 2.0):
            for names in ("both", ["both"], ["left", "right"]):
                for joint in (1, 2):
                    with self.subTest(dt=dt, names=names, joint=joint):
                        approach = make_approach(dt)
                        mapping = PhaseMap(approach.mdl.sp.phases, dt=dt)
                        if isinstance(names, list) and len(names) == 2:
                            reference = JointFaultSample(
                                *(approach.faultdomains[n] for n in names),
                                phasemaps=[mapping],
                            )
                        else:
                            reference = FaultSample(
                                approach.faultdomains["both"], mapping
                            )
                        phase = (
                            ("late",)
                            if isinstance(names, list) and len(names) == 2
                            else "late"
                        )
                        approach.add_faultsample(
                            "late",
                            "fault_phases",
                            names,
                            phase,
                            phasemap="mdl",
                            method="all",
                            n_joint=joint,
                        )
                        reference.add_fault_phases(phase, method="all", n_joint=joint)
                        self.assertEqual(
                            [s.asdict() for s in approach.scenarios()],
                            [s.asdict() for s in reference.scenarios()],
                        )
                        np.testing.assert_array_equal(
                            sorted(approach.get_times()), np.arange(2, 5) * dt
                        )
                        self.assertTrue(
                            all(s.phase == phase for s in approach.scenarios())
                        )

    def test_even_and_quadrature_sampling_respect_the_registered_timestep(self):
        for dt in (0.25, 2.0):
            for method, args in (
                ("even", (1,)),
                ("quad", ([-0.8, 0.0, 0.8], [1.0, 4.0, 1.0])),
            ):
                with self.subTest(dt=dt, method=method):
                    approach = make_approach(dt)
                    mapping = PhaseMap(approach.mdl.sp.phases, dt=dt)
                    expected = FaultSample(approach.faultdomains["both"], mapping)
                    approach.add_faultsample(
                        "late",
                        "fault_phases",
                        "both",
                        "late",
                        phasemap="mdl",
                        method=method,
                        args=args,
                    )
                    expected.add_fault_phases("late", method=method, args=args)
                    self.assertEqual(
                        [s.asdict() for s in approach.scenarios()],
                        [s.asdict() for s in expected.scenarios()],
                    )

    def test_explicit_maps_and_disabled_defaults_keep_their_identity(self):
        supplied = PhaseMap({"manual": [0.0, 1.0]}, dt=0.5)
        registry = {"mdl": supplied, "custom": supplied}
        model = make_domain(0.25).mdl
        for default in (False, True):
            with self.subTest(default=default):
                approach = SampleApproach(
                    model,
                    phasemaps=MappingProxyType(registry),
                    def_mdl_phasemap=default,
                )
                self.assertIs(approach.phasemaps["custom"], supplied)
                self.assertIs(registry["mdl"], supplied)
                self.assertEqual(supplied.dt, 0.5)
                if default:
                    self.assertIsNot(approach.phasemaps["mdl"], supplied)
                    self.assertEqual(approach.phasemaps["mdl"].dt, 0.25)
                else:
                    self.assertIs(approach.phasemaps["mdl"], supplied)
        self.assertEqual(SampleApproach(model, def_mdl_phasemap=False).phasemaps, {})

    def test_real_runs_and_fmea_keep_phase_times_and_rates(self):
        for dt in (0.5, 2.0):
            for staged in (False, True):
                with self.subTest(dt=dt, staged=staged):
                    approach = make_approach(dt)
                    model = approach.mdl
                    reference = FaultSample(
                        approach.faultdomains["both"], PhaseMap(model.sp.phases, dt=dt)
                    )
                    approach.add_faultsample(
                        "late",
                        "fault_phases",
                        "both",
                        "late",
                        method="all",
                        phasemap="mdl",
                    )
                    reference.add_fault_phases("late", method="all")
                    result, history = propagate.fault_sample(
                        model, approach, staged=staged, showprogress=False
                    )
                    expected, expected_history = propagate.fault_sample(
                        model, reference, staged=staged, showprogress=False
                    )
                    self.assertEqual(result, expected)
                    self.assertEqual(history, expected_history)
                    tables = [
                        FMEA(
                            res,
                            sample,
                            group_by=("phase", "time"),
                            average_metric=["scenario_rate"],
                            sum_metric=["xy"],
                        )
                        for res, sample in ((result, approach), (expected, reference))
                    ]
                    self.assertEqual(
                        tables[0]["average_scenario_rate"],
                        tables[1]["average_scenario_rate"],
                    )
                    self.assertEqual(tables[0]["sum_xy"], tables[1]["sum_xy"])


if __name__ == "__main__":
    unittest.main()

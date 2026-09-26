#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for fault simulations without a nominal reference.

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
import multiprocessing
import unittest
from unittest.mock import patch

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.propagate import MultiEventSimulation, Simulation
from fmdtools.sim.sample import (
    FaultDomain,
    FaultSample,
    ParameterDomain,
    ParameterSample,
)


class TestFaultSampleWithoutNominal(unittest.TestCase):
    def make_sample(self, times=(1.0, 3.0)):
        model = ExampleFunction(sp={"end_time": 5.0})
        domain = FaultDomain(model)
        domain.add_fault("examplefunction", "low")
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times(list(times))
        return model, sample

    def assert_equal_output(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def assert_direct_scenarios(self, model, sample, result, history, to_return=None):
        for scenario in sample.scenarios():
            kwargs = {} if to_return is None else {"to_return": to_return}
            expected_result, expected_history = Simulation(
                mdl=model, scen=scenario, **kwargs
            )()
            self.assert_equal_output(
                result.get(scenario.name).flatten(), expected_result
            )
            self.assert_equal_output(
                history.get(scenario.name).flatten(), expected_history
            )
        self.assertFalse(any(key.startswith("nominal.") for key in result))
        self.assertFalse(any(key.startswith("nominal.") for key in history))

    def test_public_fault_sample_runs_without_creating_a_nominal_reference(self):
        model, sample = self.make_sample()
        state = copy.deepcopy(model.s.asdict())
        initial_time = model.t.time
        with patch.object(
            MultiEventSimulation,
            "run_nom",
            side_effect=AssertionError("unexpected nominal run"),
        ):
            result, history = propagate.fault_sample(
                model, sample, staged=False, include_nominal=False, showprogress=False
            )
        self.assert_direct_scenarios(model, sample, result, history)
        self.assertEqual(model.s.asdict(), state)
        self.assertEqual(model.t.time, initial_time)

    def test_result_requests_at_intermediate_times_and_end_are_preserved(self):
        model, sample = self.make_sample()
        requested = {2.0: ["s.x", "s.y"], "end": ["classify", "s.x", "s.y"]}
        result, history = propagate.fault_sample(
            model,
            sample,
            staged=False,
            include_nominal=False,
            to_return=requested,
            showprogress=False,
        )
        self.assert_direct_scenarios(model, sample, result, history, requested)
        self.assertTrue(any("t2p0.s.x" in key for key in result))

    def test_empty_sample_returns_empty_outputs_when_nominal_is_disabled(self):
        model, sample = self.make_sample(times=())
        with patch.object(
            MultiEventSimulation,
            "run_nom",
            side_effect=AssertionError("unexpected nominal run"),
        ):
            result, history = propagate.fault_sample(
                model, sample, staged=False, include_nominal=False, showprogress=False
            )
        self.assertIsInstance(result, Result)
        self.assertIsInstance(history, History)
        self.assertEqual(dict(result), {})
        self.assertEqual(dict(history), {})

    def test_single_faults_wrapper_also_supports_disabling_nominal(self):
        model, _ = self.make_sample()
        reference, reference_history = propagate.single_faults(
            model, times=[1.0], staged=False, showprogress=False
        )
        result, history = propagate.single_faults(
            model, times=[1.0], staged=False, include_nominal=False, showprogress=False
        )
        self.assert_equal_output(
            result, {k: v for k, v in reference.items() if not k.startswith("nominal.")}
        )
        self.assert_equal_output(
            history,
            {
                k: v
                for k, v in reference_history.items()
                if not k.startswith("nominal.")
            },
        )

    def test_staging_and_requested_nominal_keep_existing_reference_data(self):
        for staged, include_nominal in ((False, True), (True, False), (True, True)):
            with self.subTest(staged=staged, include_nominal=include_nominal):
                model, sample = self.make_sample()
                simulation = MultiEventSimulation(
                    mdl=model,
                    samp=sample,
                    staged=staged,
                    include_nominal=include_nominal,
                    showprogress=False,
                )
                result, history = simulation()
                self.assertEqual(
                    any(k.startswith("nominal.") for k in result), include_nominal
                )
                self.assertEqual(
                    any(k.startswith("nominal.") for k in history), include_nominal
                )
                self.assertEqual(set(simulation.mdls), {1.0, 3.0} if staged else set())
                for scenario in sample.scenarios():
                    expected, expected_history = Simulation(mdl=model, scen=scenario)()
                    self.assert_equal_output(
                        result.get(scenario.name).flatten(), expected
                    )
                    self.assert_equal_output(
                        history.get(scenario.name).flatten(), expected_history
                    )

    def test_available_nominal_references_are_forwarded_without_copying(self):
        model, sample = self.make_sample()
        for have_result, have_history in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(result=have_result, history=have_history):
                simulation = MultiEventSimulation(
                    mdl=model,
                    samp=sample,
                    staged=False,
                    include_nominal=False,
                    showprogress=False,
                )
                nominal_result = Result({"value": 1.0})
                nominal_history = History({"time": np.array([0.0])})
                if have_result:
                    simulation.result["nominal"] = nominal_result
                if have_history:
                    simulation.history["nominal"] = nominal_history
                for _, kwargs in simulation.gen_inputs(sample.scenarios()):
                    self.assertEqual("nomresult" in kwargs, have_result)
                    self.assertEqual("nomhist" in kwargs, have_history)
                    if have_result:
                        self.assertIs(kwargs["nomresult"], nominal_result)
                    if have_history:
                        self.assertIs(kwargs["nomhist"], nominal_history)

    def test_spawn_pool_executes_the_same_fault_scenarios(self):
        model, sample = self.make_sample()
        with multiprocessing.get_context("spawn").Pool(2) as pool:
            result, history = propagate.fault_sample(
                model,
                sample,
                staged=False,
                include_nominal=False,
                pool=pool,
                auto_close_pool=False,
                showprogress=False,
            )
        self.assert_direct_scenarios(model, sample, result, history)

    def test_generated_nested_samples_still_obtain_the_required_nominal_run(self):
        model, _ = self.make_sample()
        domain = ParameterDomain(ExampleParameter)
        domain.add_variables("x", "y")
        parameters = ParameterSample(domain)
        parameters.add_variable_scenario(2.0, 3.0)
        parameters.add_variable_scenario(4.0, 1.0)
        result, history, approaches = propagate.nested_sample(
            model,
            parameters,
            staged=False,
            include_nominal=False,
            showprogress=False,
            faultdomains={"low": (("fault", "examplefunction", "low"), {})},
            faultsamples={"times": (("fault_times", "low", [1.0]), {})},
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(len(approaches), 2)
        self.assertFalse(any(".nominal." in key for key in result))
        self.assertFalse(any(".nominal." in key for key in history))
        self.assertTrue(all("examplefunction_low" in key for key in result))


if __name__ == "__main__":
    unittest.main()

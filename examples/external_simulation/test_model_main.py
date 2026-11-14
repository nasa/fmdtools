#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contract and native-lifecycle tests for the external-calculation example.

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

import pickle
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock, patch

import numpy as np
from fmdtools_examples.external_simulation.adapter import call_step
from fmdtools_examples.external_simulation.external_sim import (
    StepRequest,
    StepResult,
    reference_step,
)
from fmdtools_examples.external_simulation.model_main import CoupledFunction
from fmdtools_examples.external_simulation.pump_scenarios import sample_pump_stops

from fmdtools.sim import propagate


def alternative_step(request):
    """A second, top-level implementation with a deliberately different gain."""
    return StepResult(request.value + 3.0 * request.rate * request.dt)


class AlternativeFunction(CoupledFunction):
    external_step = staticmethod(alternative_step)


def fail_step(request):
    """Represent an execution failure, not a modeled actuator fault."""
    raise TimeoutError("external calculation failed")


class FailingFunction(CoupledFunction):
    external_step = staticmethod(fail_step)


class TestExternalAdapterPattern(unittest.TestCase):
    """Protect the boundary separately from solver-specific numerical behavior."""

    def assert_history_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            with self.subTest(history=name):
                np.testing.assert_array_equal(first[name], second[name])

    def test_reference_contract_is_explicit_immutable_and_repeatable(self):
        request = StepRequest(time=3.0, dt=0.5, value=5.0, rate=-2.0)
        with self.assertRaises(FrozenInstanceError):
            request.value = 100.0
        self.assertEqual(call_step(request), StepResult(4.0))
        self.assertEqual(call_step(request), call_step(request))
        self.assertEqual(request, StepRequest(3.0, 0.5, 5.0, -2.0))
        backend = Mock(side_effect=reference_step)
        self.assertEqual(call_step(request, backend), StepResult(4.0))
        backend.assert_called_once()
        self.assertIsNot(backend.call_args.args[0], request)
        with self.assertRaises(FrozenInstanceError):
            call_step(request).value = 100.0

    def test_invalid_requests_never_reach_the_backend(self):
        request = StepRequest(1.0, 1.0, 0.0, 2.0)
        for field in ("time", "dt", "value", "rate"):
            for bad in (float("nan"), float("inf"), -float("inf"), True, "2", [2]):
                with self.subTest(field=field, bad=bad):
                    backend = Mock(side_effect=reference_step)
                    with self.assertRaises(ValueError):
                        call_step(replace(request, **{field: bad}), backend)
                    backend.assert_not_called()
        for time, dt in ((1.0, 0.0), (1.0, -1.0), (0.0, 1.0), (-1.0, 1.0)):
            with self.subTest(time=time, dt=dt):
                backend = Mock(side_effect=reference_step)
                with self.assertRaises(ValueError):
                    call_step(replace(request, time=time, dt=dt), backend)
                backend.assert_not_called()
        with self.assertRaises(TypeError):
            call_step({"time": 1.0})
        with self.assertRaises(ValueError):
            call_step(replace(request, value=10**400))

    def test_responses_are_validated_before_any_output_is_committed(self):
        for bad in (
            7.0,
            {},
            None,
            StepResult(True),
            StepResult("7"),
            StepResult(np.array([7.0])),
            StepResult(float("nan")),
            StepResult(float("inf")),
        ):
            with self.subTest(response=bad):
                model = CoupledFunction()
                model(time=1.0)
                before = model.s.asdict()
                backend = Mock(return_value=bad)
                with patch.object(
                    CoupledFunction, "external_step", staticmethod(backend)
                ):
                    with self.assertRaises(Exception):
                        model(time=2.0)
                self.assertEqual(model.s.asdict(), before)
                self.assertFalse(model.m.any_faults())
                backend.assert_called_once()
        # Finite inputs can still overflow inside the calculation.
        with self.assertRaisesRegex(ValueError, "response.value"):
            call_step(StepRequest(1.0, 1.0, 1e308, 1e308))

    def test_execution_failures_propagate_without_retry_or_nominal_substitution(self):
        error = TimeoutError("backend timeout")
        backend = Mock(side_effect=error)
        with self.assertRaises(TimeoutError) as raised:
            call_step(StepRequest(1.0, 1.0, 0.0, 2.0), backend)
        self.assertIs(raised.exception, error)
        backend.assert_called_once()
        with self.assertRaises(Exception) as raised:
            propagate.nominal(FailingFunction())
        cause = raised.exception
        while cause.__cause__ is not None:
            cause = cause.__cause__
        self.assertIsInstance(cause, TimeoutError)
        self.assertEqual(str(cause), "external calculation failed")

    def test_native_time_steps_match_an_independent_integral(self):
        for history_dt, execution_dt in ((1.0, 1.0), (0.5, 0.5), (1.0, 0.5)):
            for rate in (2.0, 0.0, -2.0):
                with self.subTest(
                    history_dt=history_dt, execution_dt=execution_dt, rate=rate
                ):
                    model = CoupledFunction(
                        p={"rate": rate}, sp={"dt": history_dt}, t={"dt": execution_dt}
                    )
                    result, history = propagate.nominal(model)
                    np.testing.assert_allclose(
                        history["s.value"], rate * history["time"]
                    )
                    np.testing.assert_allclose(
                        history["s.calls"], history["time"] / execution_dt
                    )
                    self.assertEqual(result["tend.classify.value"], rate * 3.0)
                    self.assertEqual(model.s.value, 0.0)
                    self.assertEqual(model.s.calls, 0)
        model = CoupledFunction()
        model(time=0.0)
        self.assertEqual(model.s.calls, 0)
        model(time=1.0)
        before = model.s.asdict()
        model(time=1.0)
        self.assertEqual(model.s.asdict(), before)

    def test_native_random_inputs_are_explicit_and_seeded(self):
        for seed in (0, 31):
            with self.subTest(seed=seed):
                model = CoupledFunction(r={"seed": seed}, sp={"run_stochastic": True})
                result, history = propagate.nominal(model)
                # Native auto-update also draws at t=0; the initial value is not advanced.
                draws = np.random.default_rng(seed).uniform(0.9, 1.1, 4)
                np.testing.assert_array_equal(history["r.s.multiplier"], draws)
                expected = np.r_[0.0, np.cumsum(2.0 * draws[1:])]
                np.testing.assert_allclose(history["s.value"], expected)
                again, again_history = propagate.nominal(model)
                self.assertEqual(dict(result), dict(again))
                self.assert_history_equal(history, again_history)

    def test_fault_sampling_preserves_pairing_and_staged_equivalence(self):
        for stochastic in (False, True):
            with self.subTest(stochastic=stochastic):
                model = CoupledFunction(
                    r={"seed": 31}, sp={"run_stochastic": stochastic}
                )
                sample, result, history = sample_pump_stops(model)
                _, staged, staged_history = sample_pump_stops(model, staged=True)
                self.assertEqual(dict(result), dict(staged))
                self.assert_history_equal(history, staged_history)
                nominal = history.get("nominal")
                for scenario in sample.scenarios():
                    faulty = history.get(scenario.name)
                    np.testing.assert_array_equal(
                        faulty["r.s.multiplier"], nominal["r.s.multiplier"]
                    )
                    rates = 2.0 * nominal["r.s.multiplier"].copy()
                    rates[
                        (nominal["time"] == 0) | (nominal["time"] >= scenario.time)
                    ] = 0.0
                    np.testing.assert_allclose(faulty["s.value"], np.cumsum(rates))
                    self.assertEqual(result[scenario.name + ".tend.classify.calls"], 3)
                rates_by_time = {s.time: s.rate for s in sample.scenarios()}
                self.assertEqual(rates_by_time, {1.0: 0.0025, 2.0: 0.0075})

    def test_stop_times_must_fall_on_the_simulation_grid(self):
        for sp in ({"end_time": 1.0}, {"dt": 2.0}):
            with self.subTest(sp=sp), self.assertRaisesRegex(ValueError, "1 and 2"):
                sample_pump_stops(CoupledFunction(sp=sp))

    def test_substitution_copy_pickle_reset_and_new_keep_the_contract(self):
        for cls in (CoupledFunction, AlternativeFunction):
            for stochastic in (False, True):
                with self.subTest(cls=cls.__name__, stochastic=stochastic):
                    model = cls(r={"seed": 31}, sp={"run_stochastic": stochastic})
                    expected, expected_history = propagate.nominal(model)
                    model(time=1.0)
                    before = model.s.asdict()
                    for copied in (model.copy(), pickle.loads(pickle.dumps(model))):
                        copied(time=3.0)
                        self.assertEqual(
                            copied.classify()["value"], expected["tend.classify.value"]
                        )
                        self.assert_history_equal(copied.h.flatten(), expected_history)
                        self.assertEqual(model.s.asdict(), before)
                    model.reset()
                    model(time=3.0)
                    self.assert_history_equal(model.h.flatten(), expected_history)
                    fresh = model.new()
                    self.assertIsInstance(fresh, cls)
                    self.assertEqual(fresh.s.value, 0.0)
                    if not stochastic:
                        self.assertEqual(
                            expected["tend.classify.value"],
                            6.0 if cls is CoupledFunction else 18.0,
                        )

    def test_invalid_configuration_is_rejected_at_construction(self):
        for kwargs in (
            {"p": {"rate": float("inf")}},
            {"p": {"rate": float("nan")}},
            {"sp": {"units": "min"}},
            {"sp": {"start_time": 1.0}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(Exception):
                CoupledFunction(**kwargs)


if __name__ == "__main__":
    unittest.main()

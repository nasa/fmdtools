#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for JSON serialization of scalar and multidimensional arrays.

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

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import Function
from fmdtools.define.container.rand import Rand
from fmdtools.define.container.state import State
from fmdtools.sim.propagate import Simulation


class MatrixNoiseState(State):
    noise: np.array = np.zeros((2, 3))
    noise_update = ("uniform", (0.0, 1.0, (2, 3)))


class MatrixRand(Rand):
    s: MatrixNoiseState = MatrixNoiseState()


class MatrixFunction(Function):
    container_r = MatrixRand

    def classify(self, **kwargs):
        return {"last_noise": self.r.s.noise.copy()}


class TestJsonResultArrays(unittest.TestCase):
    def assert_same_values(self, actual, expected):
        actual, expected = actual.flatten(), expected.flatten()
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_scalar_vector_matrix_and_tensor_nesting_roundtrip(self):
        for cls in (Result, History):
            for shape in ((), (3,), (2, 3), (3, 1), (2, 1, 1), (2, 2, 2), (2, 0)):
                for dtype in (np.int64, np.float32, np.bool_):
                    with self.subTest(cls=cls.__name__, shape=shape, dtype=dtype):
                        value = np.arange(np.prod(shape, dtype=int)).reshape(shape)
                        value = value.astype(dtype)
                        original = cls({"nested.value": value})
                        before = value.copy()
                        with TemporaryDirectory() as directory:
                            path = Path(directory) / "result.json"
                            original.save(str(path))
                            self.assertEqual(
                                json.loads(path.read_text()),
                                {"nested.value": value.tolist()},
                            )
                            for renest in (False, True):
                                loaded = cls.load(str(path), renest_dict=renest)
                                self.assertIs(type(loaded), cls)
                                self.assert_same_values(loaded, original)
                            self.assertEqual(set(Path(directory).iterdir()), {path})
                        self.assertIs(original["nested.value"], value)
                        np.testing.assert_array_equal(value, before)

    def test_json_compatible_object_and_unicode_arrays(self):
        for value in (
            np.array(["alpha", "β"], dtype=object),
            np.array([["alpha"], ["β"]]),
            np.array([True, None, 7, "text"], dtype=object),
        ):
            with self.subTest(dtype=value.dtype, shape=value.shape):
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "values.json"
                    Result({"value": value}).save(str(path))
                    self.assertEqual(
                        json.loads(path.read_text()), {"value": value.tolist()}
                    )

    def test_noncontiguous_and_readonly_arrays_are_not_modified(self):
        base = np.arange(24.0).reshape(4, 6)
        for value in (base[::2, ::2], np.asfortranarray(base), base.T):
            with self.subTest(strides=value.strides):
                value.flags.writeable = False
                before = value.copy()
                with TemporaryDirectory() as directory:
                    path = Path(directory) / "output"
                    Result({"value": value}).save(str(path), filetype="json")
                    restored = Result.load(str(path), filetype="json")
                    np.testing.assert_array_equal(restored["value"], before)
                self.assertFalse(value.flags.writeable)
                np.testing.assert_array_equal(value, before)
        np.testing.assert_array_equal(base, np.arange(24.0).reshape(4, 6))

    def test_existing_one_dimensional_json_encoding_is_unchanged(self):
        values = {"z": np.array([1, 2]), "a": np.array([1.5, 2.5]), "flag": True}
        expected = json.dumps(
            {"z": [1, 2], "a": [1.5, 2.5], "flag": True},
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            Result(values).save(str(path))
            self.assertEqual(path.read_text(), expected)

    def test_individual_nested_histories_keep_the_scenario_wrapper(self):
        original = History(
            {
                "state": History({"x": np.arange(6).reshape(3, 2)}),
                "time": np.array([0.0, 1.0, 2.0]),
            }
        )
        before = copy.deepcopy(original)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            original.save(str(path), result_id="sample0")
            individual = path.parent / "history" / "sample0.json"
            payload = json.loads(individual.read_text())
            self.assertEqual(set(payload), {"sample0"})
            self.assertEqual(payload["sample0"]["state.x"], [[0, 1], [2, 3], [4, 5]])
            loaded = History.load(str(individual), indiv=True)
            self.assert_same_values(loaded, History({"sample0": original}))
            self.assertFalse(path.exists())
        self.assert_same_values(original, before)

    def test_actual_vector_random_state_simulation_saves_results_and_history(self):
        model = MatrixFunction(
            sp={"end_time": 3.0, "run_stochastic": True}, r={"seed": 31}
        )
        result, history = Simulation(mdl=model)()
        rng = np.random.default_rng(31)
        draws = np.array([rng.uniform(0.0, 1.0, (2, 3)) for _ in history.time])
        np.testing.assert_array_equal(history["r.s.noise"], draws)
        np.testing.assert_array_equal(result["tend.classify.last_noise"], draws[-1])
        with TemporaryDirectory() as directory:
            for name, output in (("result", result), ("history", history)):
                path = Path(directory) / (name + ".json")
                output.save(str(path))
                restored = type(output).load(str(path))
                self.assert_same_values(restored, output)
        np.testing.assert_array_equal(model.r.s.noise, np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()

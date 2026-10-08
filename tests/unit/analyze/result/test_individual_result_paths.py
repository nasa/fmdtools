#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for individual result filenames with dotted paths.

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

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fmdtools.analyze.common import create_indiv_filename
from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result, load
from fmdtools.define.block.function import ExampleFunction
from fmdtools.define.container.parameter import ExampleParameter
from fmdtools.sim import propagate
from fmdtools.sim.sample import ParameterDomain, ParameterSample


class TestIndividualResultPaths(unittest.TestCase):
    def assert_values(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_suffix_is_inserted_before_only_the_final_extension(self):
        cases = (
            ("result.csv", "result"),
            ("result.v1.csv", "result.v1"),
            ("runs.v1/results.csv", "runs.v1/results"),
            ("runs.v1/results.v2.json", "runs.v1/results.v2"),
            ("./results.npz", "./results"),
            ("../runs.v1/results.csv", "../runs.v1/results"),
            (".results.csv", ".results"),
        )
        for filename, stem in cases:
            for separator in ("_", "/", "__"):
                with self.subTest(filename=filename, separator=separator):
                    extension = filename[filename.rfind(".") :]
                    self.assertEqual(
                        create_indiv_filename(filename, "case.2", separator),
                        stem + separator + "case.2" + extension,
                    )

    def test_extensionless_names_do_not_gain_a_spurious_period(self):
        for filename in ("result", ".result", "runs.v1/result"):
            with self.subTest(filename=filename):
                self.assertEqual(create_indiv_filename(filename, "3"), filename + "_3")

    def test_real_result_and_history_files_roundtrip_in_dotted_directories(self):
        for filetype in ("csv", "json", "npz"):
            for result_type in (Result, History):
                with self.subTest(filetype=filetype, result_type=result_type.__name__):
                    with TemporaryDirectory() as directory:
                        root = Path(directory)
                        parent = root / "runs.v1" / ".batch"
                        parent.mkdir(parents=True)
                        filename = parent / ("results.v2." + filetype)
                        filename.write_bytes(b"untouched aggregate placeholder")
                        original = (
                            result_type(
                                {
                                    "s.value": np.array([1.0, 2.0, 3.0]),
                                    "time": np.array([0.0, 0.5, 1.0]),
                                }
                            )
                            if result_type is History
                            else result_type({"metric": 2.5, "count": 7})
                        )
                        before = {
                            key: np.copy(value) for key, value in original.items()
                        }
                        original.save(str(filename), result_id="case.2")
                        expected = parent / "results.v2" / ("case.2." + filetype)
                        self.assertTrue(expected.is_file())
                        self.assertEqual(
                            filename.read_bytes(), b"untouched aggregate placeholder"
                        )
                        self.assertEqual(
                            {p for p in root.rglob("*") if p.is_file()},
                            {filename, expected},
                        )
                        restored = load(
                            str(expected),
                            indiv=True,
                            renest_dict=False,
                            Rclass=result_type,
                        )
                        self.assert_values(
                            restored, {"case.2." + k: v for k, v in original.items()}
                        )
                        self.assert_values(original, before)

    def test_overwrite_protection_applies_to_the_correct_individual_file(self):
        for filetype in ("csv", "json", "npz"):
            with self.subTest(filetype=filetype), TemporaryDirectory() as directory:
                filename = Path(directory) / "batch.v1" / ("results.v2." + filetype)
                Result(metric=1.0).save(str(filename), result_id="case")
                target = filename.parent / "results.v2" / ("case." + filetype)
                before = target.read_bytes()
                with self.assertRaisesRegex(Exception, "File already exists"):
                    Result(metric=2.0).save(str(filename), result_id="case")
                self.assertEqual(target.read_bytes(), before)
                with redirect_stdout(StringIO()):
                    Result(metric=2.0).save(
                        str(filename), result_id="case", overwrite=True
                    )
                self.assert_values(
                    load(str(target), indiv=True, renest_dict=False),
                    {"case.metric": 2.0},
                )

    def test_parameter_simulation_saves_and_reloads_every_individual(self):
        domain = ParameterDomain(ExampleParameter)
        domain.add_variables("x", "y")
        sample = ParameterSample(domain)
        sample.add_variable_scenario(2.0, 3.0, name="run")
        sample.add_variable_scenario(4.0, 1.0, name="run")
        for filetype in ("csv", "json", "npz"):
            with self.subTest(filetype=filetype), TemporaryDirectory() as directory:
                parent = Path(directory) / "experiment.v1"
                model = ExampleFunction(sp={"end_time": 2.0})
                result, history = propagate.parameter_sample(
                    model,
                    sample,
                    showprogress=False,
                    save_indiv=True,
                    result_filename=str(parent / ("result.v2." + filetype)),
                    history_filename=str(parent / ("history.v2." + filetype)),
                )
                self.assert_values(
                    Result.load_folder(str(parent / "result.v2"), filetype), result
                )
                self.assert_values(
                    History.load_folder(str(parent / "history.v2"), filetype), history
                )
                expected_paths = {
                    parent / folder / (scenario.name + "." + filetype)
                    for folder in ("result.v2", "history.v2")
                    for scenario in sample.scenarios()
                }
                self.assertEqual(
                    {p for p in parent.rglob("*") if p.is_file()}, expected_paths
                )


if __name__ == "__main__":
    unittest.main()

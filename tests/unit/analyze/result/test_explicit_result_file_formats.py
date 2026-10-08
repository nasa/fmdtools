#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for explicitly selected result file formats.

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
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout

import numpy as np

from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result, load
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim.propagate import Simulation


class TestExplicitResultFileFormats(unittest.TestCase):
    def assert_same(self, actual, expected):
        actual, expected = actual.flatten(), expected.flatten()
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def examples(self):
        return (
            Result({"metric": 3.5, "nested.value": 8.25}),
            History(
                {
                    "time": np.array([0.0, 1.0, 2.0]),
                    "s.value": np.array([3.0, 4.0, 5.0]),
                }
            ),
        )

    def test_explicit_formats_write_only_the_requested_file_and_roundtrip(self):
        for original in self.examples():
            for filetype in ("csv", "json", "npz"):
                for suffix in ("", ".data", ".csv", ".json", ".npz"):
                    with (
                        self.subTest(
                            cls=type(original).__name__,
                            filetype=filetype,
                            suffix=suffix,
                        ),
                        TemporaryDirectory() as directory,
                    ):
                        before = copy.deepcopy(original)
                        path = Path(directory) / ("output" + suffix)
                        original.save(str(path), filetype=filetype)
                        self.assertEqual(set(Path(directory).iterdir()), {path})
                        self.assertGreater(path.stat().st_size, 0)
                        self.assert_encoding(path, original, filetype)
                        for nested in (False, True):
                            restored = type(original).load(
                                str(path), filetype=filetype, renest_dict=nested
                            )
                            self.assertIs(type(restored), type(original))
                            self.assert_same(restored, original)
                        self.assert_same(original, before)

    def assert_encoding(self, path, original, filetype):
        if filetype == "npz":
            with np.load(path, allow_pickle=False) as archive:
                self.assert_same(
                    Result({key: archive[key] for key in archive.files}), original
                )
        elif filetype == "json":
            self.assert_same(Result(json.loads(path.read_text())), original)
        else:
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], list(original.keys()))
            expected_rows = 3 if isinstance(original, History) else 1
            self.assertEqual(len(rows), expected_rows + 1)

    def test_result_load_forwards_format_for_independently_written_files(self):
        for filetype in ("csv", "json", "npz"):
            with self.subTest(filetype=filetype), TemporaryDirectory() as directory:
                path = Path(directory) / "external"
                expected = Result({"metric": 4.5})
                if filetype == "csv":
                    path.write_text("metric\n4.5\n")
                elif filetype == "json":
                    path.write_text('{"metric": 4.5}')
                else:
                    with path.open("wb") as stream:
                        np.savez(stream, metric=4.5)
                before = path.read_bytes()
                self.assert_same(load(str(path), filetype=filetype), expected)
                self.assert_same(Result.load(str(path), filetype=filetype), expected)
                self.assertEqual(path.read_bytes(), before)

    def test_npz_does_not_overwrite_a_neighbor_with_an_appended_extension(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result"
            neighbor = Path(directory) / "result.npz"
            neighbor.write_bytes(b"unrelated saved data")
            Result({"metric": 2.0}).save(str(path), filetype="npz")
            self.assertEqual(neighbor.read_bytes(), b"unrelated saved data")
            self.assert_same(
                Result.load(str(path), filetype="npz"), Result({"metric": 2.0})
            )

    def test_overwrite_policy_remains_effective_for_each_explicit_format(self):
        for filetype in ("csv", "json", "npz"):
            with self.subTest(filetype=filetype), TemporaryDirectory() as directory:
                path = Path(directory) / "result"
                path.write_bytes(b"existing content")
                with self.assertRaisesRegex(Exception, "already exists"):
                    Result({"metric": 2.0}).save(str(path), filetype=filetype)
                self.assertEqual(path.read_bytes(), b"existing content")
                with redirect_stdout(io.StringIO()):
                    Result({"metric": 7.0}).save(
                        str(path), filetype=filetype, overwrite=True
                    )
                self.assertEqual(set(Path(directory).iterdir()), {path})
                self.assert_same(
                    Result.load(str(path), filetype=filetype), Result({"metric": 7.0})
                )

    def test_inferred_extensions_and_individual_result_layout_are_unchanged(self):
        for filetype in ("csv", "json", "npz"):
            for original in self.examples():
                with (
                    self.subTest(filetype=filetype, cls=type(original).__name__),
                    TemporaryDirectory() as directory,
                ):
                    path = Path(directory) / ("result." + filetype)
                    original.save(str(path))
                    self.assert_same(type(original).load(str(path)), original)
                    original.save(str(path), result_id="scen0")
                    individual = Path(directory) / "result" / ("scen0." + filetype)
                    restored = type(original).load(str(individual), indiv=True)
                    self.assert_same(
                        restored, type(original)({"scen0": original}).flatten()
                    )
                    self.assertGreater(path.stat().st_size, 0)

    def test_real_simulation_outputs_roundtrip_without_filename_extensions(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        result, history = Simulation(mdl=model)()
        for filetype in ("csv", "json", "npz"):
            with self.subTest(filetype=filetype), TemporaryDirectory() as directory:
                for name, output in (("result", result), ("history", history)):
                    path = Path(directory) / name
                    output.save(str(path), filetype=filetype)
                    self.assert_same(
                        type(output).load(str(path), filetype=filetype), output
                    )
                self.assertEqual(
                    {p.name for p in Path(directory).iterdir()}, {"result", "history"}
                )


if __name__ == "__main__":
    unittest.main()

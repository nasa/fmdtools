#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for selecting result files from mixed-content folders.

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
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from fmdtools.analyze.common import load_folder
from fmdtools.analyze.history import History
from fmdtools.analyze.result import Result
from fmdtools.define.block.function import ExampleFunction
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


def add_unrelated_entries(folder, filetype):
    for name in (
        ".DS_Store",
        "README",
        "README.md",
        "notes.txt",
        "backup." + filetype + ".bak",
        "upper." + filetype.upper(),
    ):
        (folder / name).write_bytes(b"unrelated content")
    for other in ("csv", "json", "npz"):
        if other != filetype:
            (folder / ("other." + other)).write_bytes(b"not selected")
    (folder / ("subfolder." + filetype)).mkdir()
    (folder / "nested").mkdir()
    (folder / "nested" / ("inner." + filetype)).write_bytes(b"not recursive")


class TestResultFolderSelection(unittest.TestCase):
    def assert_same(self, actual, expected):
        actual, expected = actual.flatten(), expected.flatten()
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_selects_exact_extensions_and_files_in_existing_directory_order(self):
        for filetype in ("csv", "json", "npz"):
            with self.subTest(filetype=filetype), TemporaryDirectory() as directory:
                folder = Path(directory)
                names = {
                    "z." + filetype,
                    "run.v2." + filetype,
                    ".hidden." + filetype,
                    "." + filetype,
                }
                for name in names:
                    (folder / name).write_bytes(b"selected file")
                add_unrelated_entries(folder, filetype)
                before = {
                    path.name: path.read_bytes()
                    for path in folder.iterdir()
                    if path.is_file()
                }
                expected = [name for name in os.listdir(folder) if name in names]
                self.assertEqual(load_folder(str(folder), filetype), expected)
                self.assertEqual(load_folder(folder, filetype), expected)
                self.assertEqual(
                    before,
                    {
                        path.name: path.read_bytes()
                        for path in folder.iterdir()
                        if path.is_file()
                    },
                )

    def test_public_loaders_roundtrip_individual_files_among_unrelated_entries(self):
        examples = (
            Result({"metric": 2.5, "end.count": 3}),
            History(
                {
                    "time": np.array([0.0, 1.0, 2.0]),
                    "s.value": np.array([3.0, 5.0, 8.0]),
                }
            ),
        )
        for output in examples:
            for filetype in ("csv", "json", "npz"):
                with (
                    self.subTest(cls=type(output).__name__, filetype=filetype),
                    TemporaryDirectory() as directory,
                ):
                    target = Path(directory) / ("results.v2." + filetype)
                    original = copy.deepcopy(output)
                    for name in ("nominal", "fault_1"):
                        output.save(str(target), result_id=name)
                    folder = target.with_suffix("")
                    add_unrelated_entries(folder, filetype)
                    expected = type(output)({"nominal": output, "fault_1": output})
                    for nested in (False, True):
                        actual = type(output).load_folder(
                            str(folder), filetype, renest_dict=nested
                        )
                        self.assertIs(type(actual), type(output))
                        self.assert_same(actual, expected)
                    self.assert_same(output, original)

    def test_empty_or_unrelated_only_folders_return_empty_results(self):
        for filetype in ("csv", "json", "npz"):
            for unrelated in (False, True):
                with (
                    self.subTest(filetype=filetype, unrelated=unrelated),
                    TemporaryDirectory() as directory,
                ):
                    folder = Path(directory)
                    if unrelated:
                        add_unrelated_entries(folder, filetype)
                    self.assertEqual(load_folder(str(folder), filetype), [])
                    for cls in (Result, History):
                        actual = cls.load_folder(str(folder), filetype)
                        self.assertIs(type(actual), cls)
                        self.assertEqual(dict(actual), {})

    def test_errors_in_selected_files_are_not_suppressed(self):
        for cls in (Result, History):
            with self.subTest(cls=cls.__name__), TemporaryDirectory() as directory:
                path = Path(directory) / "damaged.json"
                path.write_bytes(b"{ not valid JSON")
                before = path.read_bytes()
                with self.assertRaises(json.JSONDecodeError):
                    cls.load_folder(directory, "json")
                self.assertEqual(path.read_bytes(), before)

    def test_missing_folders_and_non_directory_paths_still_raise(self):
        with TemporaryDirectory() as directory:
            ordinary = Path(directory) / "file.json"
            ordinary.write_bytes(b"{}")
            for path, error in (
                (Path(directory) / "missing", FileNotFoundError),
                (ordinary, NotADirectoryError),
            ):
                with self.subTest(path=path):
                    for loader in (
                        load_folder,
                        Result.load_folder,
                        History.load_folder,
                    ):
                        with self.assertRaises(error):
                            loader(str(path), "json")

    def test_actual_fault_simulation_outputs_load_from_mixed_folders(self):
        model = ExampleFunction(sp={"end_time": 3.0})
        domain = FaultDomain(model)
        domain.add_fault("examplefunction", "low")
        sample = FaultSample(domain, def_mdl_phasemap=False)
        sample.add_fault_times([1.0, 2.0])
        result, history = propagate.fault_sample(model, sample, showprogress=False)
        for output in (result, history):
            scenarios = output.nest(1)
            self.assertEqual(len(scenarios), 3)
            for filetype in ("csv", "json", "npz"):
                with (
                    self.subTest(cls=type(output).__name__, filetype=filetype),
                    TemporaryDirectory() as directory,
                ):
                    target = Path(directory) / ("runs." + filetype)
                    for name, scenario in scenarios.items():
                        scenario.save(str(target), result_id=name)
                    folder = target.with_suffix("")
                    add_unrelated_entries(folder, filetype)
                    restored = type(output).load_folder(str(folder), filetype)
                    self.assert_same(restored, output)


if __name__ == "__main__":
    unittest.main()

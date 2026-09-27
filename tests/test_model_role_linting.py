#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for opt-in static model-role linting.

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

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRELUDE = """
from fmdtools.define.block.function import Function
from fmdtools.define.block.component import Component
from fmdtools.define.container.state import State
from fmdtools.define.flow.base import Flow
from fmdtools.define.architecture.component import ComponentArchitecture

class ModelState(State):
    speed: float = 1.0
    active: bool = False

class Water(Flow):
    container_s = ModelState
"""


@unittest.skipUnless(
    importlib.util.find_spec("pylint"),
    "Install fmdtools[lint] to test the Pylint plugin",
)
class TestModelRoleLinting(unittest.TestCase):
    def lint(self, source, *, plugin=True, extra_files=None, profile=False):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "model.py"
            path.write_text(textwrap.dedent(source), encoding="utf-8")
            for name, text in (extra_files or {}).items():
                (directory / name).write_text(textwrap.dedent(text), encoding="utf-8")
            before = {entry.name: entry.read_bytes() for entry in directory.iterdir()}
            rcfile = directory / "empty.rc"
            rcfile.write_text("", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "pylint",
                "--output-format=json",
                "--persistent=n",
                "--score=n",
                "--reports=n",
                "--rcfile="
                + str(ROOT / "tools/pylint-models.rc" if profile else rcfile),
            ]
            if not profile:
                command += [
                    "--disable=all",
                    "--enable=no-member"
                    + (",fmdtools-unknown-field" if plugin else ""),
                ]
                if plugin:
                    command += ["--load-plugins=fmdtools_pylint"]
            command += [str(path)]
            env = dict(os.environ, PYLINTHOME=str(directory / "pylint-cache"))
            env["PYTHONPATH"] = os.pathsep.join(
                [str(ROOT / "src"), str(directory), env.get("PYTHONPATH", "")]
            )
            result = subprocess.run(
                command,
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            self.assertIn(result.returncode, (0, 2), result.stdout + result.stderr)
            self.assertEqual(result.stderr, "")
            messages = json.loads(result.stdout)
            for message in messages:
                self.assertIn(
                    message["symbol"],
                    ("no-member", "fmdtools-unknown-field"),
                    result.stdout,
                )
                self.assertEqual(Path(message["path"]).name, "model.py")
                message["path"] = "model.py"
                self.assertGreater(message["line"], 0)
            for name, contents in before.items():
                self.assertEqual((directory / name).read_bytes(), contents)
            self.assertFalse(
                (directory / "executed").exists(), "Linting executed the model"
            )
            return messages

    def test_valid_roles_resolve_without_the_baseline_false_positives(self):
        code = (
            PRELUDE
            + """
class Pump(Function):
    container_s = ModelState
    flow_water = Water
    def static_behavior(self):
        self.water.s.speed += self.s.speed
        self.s.inc(speed=1.0)
        return self.water.s.active
"""
        )
        baseline = self.lint(code, plugin=False)
        self.assertGreater(len(baseline), 0)
        self.assertEqual(self.lint(code), [])

    def test_misspelled_fields_flows_and_self_attributes_are_reported(self):
        code = (
            PRELUDE
            + """
class Pump(Function):
    container_s = ModelState
    flow_water = Water
    def static_behavior(self):
        return self.s.speeed, self.water.s.speeed, self.watre, self.unknown
"""
        )
        messages = self.lint(code)
        self.assertEqual(
            [m["symbol"] for m in messages].count("fmdtools-unknown-field"), 2
        )
        self.assertEqual([m["symbol"] for m in messages].count("no-member"), 2)
        self.assertEqual(len(messages), 4)

    def test_inherited_roles_and_overridden_state_types_remain_separate(self):
        code = (
            PRELUDE
            + """
class AlternateState(State):
    temperature: float = 10.0
class Parent(Function):
    container_s = ModelState
    flow_water = Water
class Child(Parent):
    container_s = AlternateState
    def static_behavior(self):
        self.s.temperature += self.water.s.speed
        return self.s.speed
class Sibling(Parent):
    def static_behavior(self):
        return self.s.speed
"""
        )
        messages = self.lint(code)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["symbol"], "fmdtools-unknown-field")
        self.assertIn("AlternateState", messages[0]["message"])

    def test_imported_aliased_containers_and_flows_are_inferred_without_execution(self):
        other = (
            PRELUDE
            + """
from pathlib import Path
Path('executed').write_text('must not run')
"""
        )
        code = """
from fmdtools.define.block.function import Function as F
from other import Water as W, ModelState as S
class Pump(F):
    container_s = S
    flow_water = W
    def static_behavior(self):
        return self.s.speed, self.water.s.misspelled
raise RuntimeError('must not execute model')
"""
        messages = self.lint(code, extra_files={"other.py": other})
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["symbol"], "fmdtools-unknown-field")

    def test_component_and_architecture_roles_are_inferred(self):
        code = (
            PRELUDE
            + """
class Controller(Component):
    container_s = ModelState
    flow_water = Water
    def dynamic_behavior(self):
        return self.s.active, self.water.s.speed
class Components(ComponentArchitecture):
    def marker(self):
        return 1
class Pump(Function):
    arch_ca = Components
    def static_behavior(self):
        return self.ca.marker(), self.ca.missing_method()
"""
        )
        messages = self.lint(code)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["symbol"], "no-member")
        self.assertIn("missing_method", messages[0]["message"])

    def test_unrelated_classes_with_similar_names_are_not_modified(self):
        code = (
            PRELUDE
            + """
class Function:
    pass
class Ordinary(Function):
    container_s = ModelState
    def method(self):
        return self.s.speed
"""
        )
        messages = self.lint(code)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["symbol"], "no-member")
        self.assertIn("'s' member", messages[0]["message"])

    def test_explicit_attributes_properties_and_unknown_factories_are_preserved(self):
        code = (
            PRELUDE
            + """
from unavailable_factory import create_state
class Explicit(Function):
    container_s = ModelState
    s = 1.0
    def static_behavior(self):
        return self.s.real
class WithProperty(Function):
    container_s = ModelState
    @property
    def s(self):
        return 1.0
class Child(WithProperty):
    def static_behavior(self):
        return self.s.real
class Dynamic(Function):
    container_s = create_state
    def static_behavior(self):
        return self.s.not_statically_known
"""
        )
        self.assertEqual(self.lint(code), [])

    def test_custom_role_names_and_declared_state_members_are_resolved(self):
        code = (
            PRELUDE
            + """
from fmdtools.define.object.base import BaseObject
class Extensible(BaseObject):
    roletypes = ('sensor',)
    sensor_primary = Water
    def read(self):
        return self.primary.s.speed, self.primary.s.unknown_field
"""
        )
        messages = self.lint(code)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["symbol"], "fmdtools-unknown-field")

    def test_dynamic_getattr_is_not_treated_as_a_closed_schema(self):
        code = (
            PRELUDE
            + """
class DynamicState(ModelState):
    def __getattr__(self, name):
        return 1.0
class Pump(Function):
    container_s = DynamicState
    def static_behavior(self):
        return self.s.created_dynamically
"""
        )
        self.assertEqual(self.lint(code), [])

    def test_declared_native_dunder_and_container_methods_remain_accessible(self):
        code = (
            PRELUDE
            + """
class Pump(Function):
    container_s = ModelState
    def static_behavior(self):
        return self.s.__fields__, self.s.asdict(), self.s.speed
"""
        )
        self.assertEqual(self.lint(code), [])

    def test_unknown_fields_in_write_and_delete_operations_are_reported(self):
        code = (
            PRELUDE
            + """
class Pump(Function):
    container_s = ModelState
    def static_behavior(self):
        self.s.speeed = 1.0
        del self.s.missing_field
"""
        )
        messages = self.lint(code)
        self.assertEqual(len(messages), 2)
        self.assertEqual({m["symbol"] for m in messages}, {"fmdtools-unknown-field"})

    def test_external_instances_and_annotation_only_fields_are_supported(self):
        code = (
            PRELUDE
            + """
class TypedState(State):
    speed: float
class Pump(Function):
    container_s = TypedState
pump = Pump()
state = pump.s
value = state.speed
bad = state.speeed
"""
        )
        messages = self.lint(code)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["symbol"], "fmdtools-unknown-field")

    def test_an_unknown_mixin_does_not_create_a_false_missing_field(self):
        code = (
            PRELUDE
            + """
from unavailable_extension import Mixin
class ExtendedState(Mixin, ModelState):
    pass
class Pump(Function):
    container_s = ExtendedState
    def static_behavior(self):
        return self.s.member_from_unknown_mixin
"""
        )
        self.assertEqual(self.lint(code), [])

    def test_optional_profile_produces_the_same_diagnostics(self):
        code = (
            PRELUDE
            + """
class Pump(Function):
    container_s = ModelState
    def static_behavior(self):
        return self.s.unknown_field
"""
        )
        self.assertEqual(self.lint(code, profile=True), self.lint(code))

    def test_plugin_import_does_not_import_fmdtools_or_require_simulation(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, fmdtools_pylint; assert 'fmdtools' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

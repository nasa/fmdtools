Model linting Extra
===================

What the plugin checks
----------------------

The optional Pylint plugin resolves model roles declared in fmdtools classes.
It understands statically declared containers, flows and architectures, including
inherited declarations, overridden role types, imported aliases and custom role
names. This removes spurious E1101 (no-member) messages for declared roles,
while retaining E1101 for misspelled object attributes and flow names.

It also recognizes dictionaries created by Architecture.init_flexible_roles:
flows, fxns, comps, acts and other statically declared flexible roles. These
members are inherited by model subclasses. It does not suppress typos such
as flwos or fxnss, or invent such attributes on ordinary Python classes.

E9901 (fmdtools-unknown-field) reports an unknown field on a statically
resolved container, including reads, writes and deletes. For example::

    from fmdtools.define.block.function import Function
    from fmdtools.define.container.state import State
    from fmdtools.define.flow.base import Flow

    class PumpState(State):
        pressure: float = 1.0

    class Water(Flow):
        container_s = PumpState

    class Pump(Function):
        container_s = PumpState
        flow_water = Water

        def static_behavior(self):
            self.water.s.pressure += self.s.pressure  # Known field.
            return self.water.s.presure              # E9901: typo.

Linting is mainly helpful for avoiding common syntax errors; it is not
a guarantee that a model has implemented correct behavior or is being
used correctly.

Furthermore, the fmdtools plugin is intrinsically limited to providing
basic functionality and thus may not extend to your user-defined classes.

Installation
------------

The `fmdtools_pylint` plugin is installed when when the user selects the
`[lint]` or `[all]` configuration at installation, e.g.,::

    uv pip install fmdtools[lint]

or::

    uv pip install fmdtools[all]

or, from the fmdtools repo itself::

    uv pip install .[lint]

If already using a uv-managed project environment, `uv sync --extra lint` can also
be used. 

Run a focused model-member check::

    .venv/bin/python -m pylint --rcfile=tools/pylint-models.rc examples/water_pump/model_stochastic.py # for mac/linux
    .venv/Scripts/python -m pylint --rcfile=tools/pylint-models.rc examples/water_pump/model_stochastic.py  # for Windows


That profile enables only E1101 and E9901, and should thus not return anything for this file. For the project's full Pylint
checks, run::

    .venv/bin/python -m pylint examples/water_pump/model_stochastic.py  # for mac/linux
    .venv/Scripts/python -m pylint examples/water_pump/model_stochastic.py  # for Windows

VSCode Usage
------------

To use with VSCode, install the Microsoft Python and Pylint extensions. Select the project .venv
in Python: Select Interpreter. 

Add the following code to your repository `.vscode/settings.json` file (you can also copy/use the one in the fmdtools repo):

.. code-block:: json

    {
        "pylint.cwd": "${workspaceFolder}",
        "pylint.importStrategy": "fromEnvironment",
        "pylint.path": ["${interpreter}", "-m", "pylint"]
    }

This uses Pylint from the selected interpreter and loads the .pylintrc from
the project root, avoiding the extension's otherwise possible fallback to
its bundled Pylint. No extra plugin argument is required.

To use Pylint, restart the server after changing the environment or settings.
Diagnostics should appear in the Problems panel. If they are missing, open
Output > Pylint to inspect the interpreter, command and import errors.
A terminal using a different Python does not verify the extension's setup.

Spyder Usage
------------

While Spyder does have a built-in linting support via Pylint, it is not currently able to load custom plugins (see [Issue #26423](https://github.com/spyder-ide/spyder/issues/26423)).

Thus, until this bug is fixed, we recommend running Pylint directly from the ipython console (or your script), using::

    import pylint
    pylint.run_pylint(["filename.py"])

Troubleshooting and test coverage
---------------------------------

- "unable to load plugin fmdtools_lint" uses the wrong module name:
  configure fmdtools_pylint.
- "unable to load plugin fmdtools_pylint" usually indicates that the
  analysis interpreter differs from the one where the plugin was installed.
  Verify sys.executable and the package paths in the **same** process.
- If a subclass of FunctionArchitecture still gets an E1101 for flows/fxns,
  verify the Pylint version and loaded plugin. The tests explicitly cover
  inherited architecture dictionaries and keep typos on those and ordinary
  classes detectable.

To test, you can run the integration tests in the project environment::

    .venv/bin/python -m pytest -o addopts='' tests/test_model_role_linting.py

These tests launch Pylint subprocesses, verify JSON diagnostics, check that the
source code is not executed and compare the project-root config with the
focused profile.

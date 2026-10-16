Model linting (optional)
========================

What the plugin checks
----------------------

The optional Pylint plugin resolves model roles declared in fmdtools classes.
It understands statically declared containers, flows and architectures, including
inherited declarations, overridden role types, imported aliases and custom role
names. This removes spurious E1101 (no-member) messages for declared roles,
while retaining E1101 for misspelled object attributes and flow names.

It also recognises dictionaries created by Architecture.init_flexible_roles:
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

The plugin does not check simulation results or physical equations.
It does not implement semantic colouring, autocomplete, or unresolved-flow
highlighting (the remaining parts of issue #14). Constructor-supplied roles,
dynamic connection dictionaries and role names that cannot be inferred
statically remain outside its scope. Properties, dynamic getattr methods and
unknown mixins retain conservative Pylint behaviour. Source files and imported
model helpers are parsed; their top-level statements are never executed.

Installation: use the same Python as Pylint
-------------------------------------------

Install Pylint and this plugin in the same interpreter that actually performs
analysis. Installing them only in an editor's interactive console does not
necessarily install them in the editor's separate analysis process.

From the repository root, using uv (https://docs.astral.sh/uv/)::

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -e ".[lint]"
    .venv/bin/python -m pylint --version
    .venv/bin/python -c "import sys, pylint, fmdtools_pylint; print(sys.executable); print(pylint.__file__); print(fmdtools_pylint.__file__)"

On Windows, replace .venv/bin/python with .venv/Scripts/python.exe.
If already using a uv-managed project environment, uv sync --extra lint is
another option. The plugin is packaged in tools/fmdtools_pylint but imports
under the single module name fmdtools_pylint (not fmdtools_lint).

Run a focused model-member check::

    .venv/bin/python -m pylint --rcfile=tools/pylint-models.rc examples/water_pump/model_stochastic.py

That profile enables only E1101 and E9901. For the project's regular Pylint
checks, run without the focused profile::

    .venv/bin/python -m pylint examples/water_pump/model_stochastic.py

The repository-root .pylintrc loads the plugin without disabling normal
Pylint checks. tools/pylint-models.rc is a separate, focused profile.
The config file does not install modules by itself; the Pylint process
must import fmdtools_pylint successfully first.

VS Code
-------

Install the Microsoft Python and Pylint extensions. Select the project .venv
in Python: Select Interpreter. The committed .vscode/settings.json preserves
the repository's other settings and adds these keys:

.. code-block:: json

    {
        "pylint.cwd": "${workspaceFolder}",
        "pylint.importStrategy": "fromEnvironment",
        "pylint.path": ["${interpreter}", "-m", "pylint"]
    }

This uses Pylint from the selected interpreter and loads the .pylintrc from
the project root, avoiding the extension's otherwise possible fallback to
its bundled Pylint. No extra plugin argument is required.

Use Pylint: Restart Server after changing the environment or settings.
Diagnostics appear in the Problems panel. If they are missing, open
Output > Pylint to inspect the interpreter, command and import errors.
A terminal using a different Python does not verify the extension's setup.

Spyder: Code Analysis and interpreter differences
-------------------------------------------------

Open the panel from View > Panes > Code Analysis and press F8 (or use
Source > Run code analysis). Click the panel's Output button if it fails.

Spyder Code Analysis runs Pylint in the Spyder **application's analysis
environment**, which can differ from the selected IPython Console interpreter.
Changing the console interpreter or running pylint.run_pylint in that console
does not configure the built-in Code Analysis backend.

To use that pane, install the plugin in the interpreter actually running its
Pylint backend, then verify both imports **from that interpreter**::

    ANALYSIS_PYTHON -c "import sys, pylint, fmdtools_pylint; print(sys.executable); print(pylint.__file__); print(fmdtools_pylint.__file__)"
    ANALYSIS_PYTHON -m pylint --rcfile=/absolute/path/to/fmdtools/.pylintrc /absolute/path/to/model_main.py

Replace ANALYSIS_PYTHON with the Spyder application's Python executable, not
automatically the console's interpreter. The absolute configuration path
avoids depending on the current working directory. The root .pylintrc
can also be merged into an existing personal Pylint configuration used
by Spyder (do not overwrite other settings). Spyder's documentation explains
its configuration conventions:
https://docs.spyder-ide.org/current/panes/pylint.html

Some standalone Spyder installs do not allow additional packages in the
application environment or have no supported setting for selecting the
backend interpreter. In those cases, Code Analysis cannot be guaranteed
to load this plugin just by selecting a console venv. Use the verified
project-venv command-line Pylint instead; it provides the same diagnostics.
Actual Spyder GUI integration still needs testing on each installation.

Troubleshooting and test coverage
---------------------------------

- "unable to load plugin fmdtools_lint" uses the wrong module name:
  configure fmdtools_pylint.
- "unable to load plugin fmdtools_pylint" usually indicates that the
  analysis interpreter differs from the one where the plugin was installed.
  Verify sys.executable and the package paths in the **same** process.
- Reload or restart the linter after interpreter changes. Spyder IPython
  console reloads do not necessarily restart the Code Analysis subprocess.
- If a subclass of FunctionArchitecture still gets an E1101 for flows/fxns,
  verify the Pylint version and loaded plugin. The tests explicitly cover
  inherited architecture dictionaries and keep typos on those and ordinary
  classes detectable.

Run the integration tests in the project environment::

    .venv/bin/python -m pytest -o addopts='' tests/test_model_role_linting.py

The tests launch Pylint subprocesses, verify JSON diagnostics, check that the
source code is not executed and compare the project-root config with the
focused profile. Desktop editor testing is separate.

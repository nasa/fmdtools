Model linting
=============

Install and run
---------------

The optional Pylint plugin understands class-declared fmdtools roles such as
``container_s``, ``flow_water`` and ``arch_ca``. It resolves their corresponding
instance attributes without importing or running the model under inspection.
Install it in the Python environment used by the linter::

    python -m pip install -e ".[lint]"
    python -m pylint --load-plugins=fmdtools_pylint model.py

For a focused model-member check from a repository checkout::

    python -m pylint --rcfile=tools/pylint-models.rc model.py

This optional profile enables ``no-member`` and ``fmdtools-unknown-field`` only.
It does not replace your normal Pylint configuration or the repository's other
checks. To keep your existing checks, add ``fmdtools_pylint`` to your existing
``load-plugins`` setting instead of using the focused profile.

What is checked
---------------

For example::

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
            self.water.s.pressure += self.s.pressure  # Declared members.
            return self.water.s.presure              # E9901: misspelled field.

``E9901 / fmdtools-unknown-field`` reports missing members of statically resolved
containers. It also checks field writes and deletes. Pylint's normal
``E1101 / no-member`` handles unknown object attributes and misspelled flow names.
Inherited role declarations, overridden role classes and import aliases are
supported. Ordinary classes that merely use similarly named attributes are not
transformed. Explicit attributes and properties retain their ordinary inference.

The plugin is a separate ``fmdtools_pylint`` package so loading it does not import
the simulation library or load recordclass's native extension for inference.
Pylint is optional and is not imported by normal fmdtools use.

Visual Studio Code
------------------

Install Microsoft's Pylint extension, select the environment containing the lint
extra, and add the following workspace settings::

    {
        "pylint.importStrategy": "fromEnvironment",
        "pylint.args": ["--load-plugins=fmdtools_pylint"]
    }

Diagnostics appear as editor underlines and in the Problems panel. Consult the
`VS Code linting documentation <https://code.visualstudio.com/docs/python/linting>`_
for environment and extension settings. Do not use the retired
``python.linting.pylintArgs`` setting.

Spyder
------

Install the lint extra in the environment running Spyder's Pylint Code Analysis
backend, which may differ from the selected IPython-console interpreter. Add the
plugin to the existing Pylint configuration used by that backend::

    [MAIN]
    load-plugins=fmdtools_pylint

Run Code Analysis with F8. Its output provides source positions for navigation.
Spyder's `Code Analysis documentation
<https://docs.spyder-ide.org/current/panes/pylint.html>`_ describes configuration
through ``.pylintrc`` and viewing analyzer output. Do not overwrite an existing
configuration file; merge the plugin setting with any other plugins.

Limits and validation
---------------------

This provides the linting portion of issue #14, not semantic colour themes,
Pylance completion or grey highlighting of unresolved flows. Runtime-generated
connections, constructor-supplied role replacements and flexible architecture
dictionaries are not reconstructed. Unresolved factories and unknown mixins
remain uninferred; custom dynamic ``__getattr__`` implementations are respected.
Native private/dunder container members are left to the existing analyzer.

Pylint 4 is supported. Run the integration tests after installing the lint extra::

    python -m pytest -o addopts='' --testtype=custom tests/test_model_role_linting.py

These tests launch the real Pylint command, inspect JSON diagnostics and verify
that model files and imported model helpers are not executed. The desktop
interfaces themselves require separate interactive validation.

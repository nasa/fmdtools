Optional model linting
======================

Static model-member checks are available through the independently maintained
`fmdtools-pylint <https://github.com/sylvesterkaczmarek/fmdtools-pylint>`_
package, maintained by
`Sylvester Kaczmarek <https://github.com/sylvesterkaczmarek>`_.
The plugin originated in `fmdtools PR #79
<https://github.com/nasa/fmdtools/pull/79>`_ and is now distributed separately.

It recognizes declared states, flows and architecture roles, including inherited
roles, while retaining diagnostics for misspelled members. It does not validate
physical equations, simulation results or overall model correctness, and does
not provide semantic colouring or autocomplete.

Installation and editor setup
-----------------------------

Install the external plugin separately in the Python environment used for
linting, alongside your fmdtools installation. It is not bundled with fmdtools
and is not installed by ``fmdtools[all]``. See the external project's
`installation instructions
<https://github.com/sylvesterkaczmarek/fmdtools-pylint#quick-start>`_
and `VS Code / Spyder guide
<https://github.com/sylvesterkaczmarek/fmdtools-pylint/blob/main/docs/editors.md>`_
for the supported commands and compatibility information.

Once installed, opt in explicitly::

    python -m pylint --load-plugins=fmdtools_pylint path/to/model.py

For project-wide use, add ``load-plugins=fmdtools_pylint`` to your own Pylint
configuration. Preserve any other settings and enable the plugin only where it
is installed. For Spyder, the external guide describes a project-console
workflow rather than modifying Spyder's application environment.

Licensing and support
---------------------

The external plugin has its own licence, releases and
`issue tracker <https://github.com/sylvesterkaczmarek/fmdtools-pylint/issues>`_.
See its `licensing notes
<https://github.com/sylvesterkaczmarek/fmdtools-pylint/blob/main/docs/licensing.md>`_.
The fmdtools licence is unchanged. This informational link does not bundle the
external code or add a runtime or installation dependency.

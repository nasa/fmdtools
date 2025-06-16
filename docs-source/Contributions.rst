Contributions
=============


.. include:: ../CONTRIBUTORS.md
   :parser: myst_parser.sphinx_

.. include:: ../PUBLICATIONS.md
   :parser: myst_parser.sphinx_


How to Contribute
^^^^^^^^^^^^^^^^^

Development of fmdtools is coordinated by the :doc:`fmdtools team <../CONTRIBUTORS>` at NASA Ames Research Center. As an NASA-developed and maintained open-source tool, outside contributions are welcomed. To be able to submit contributions (e.g., pull requests), external contributors should first submit a contributors license agreement (`Individual CLA <https://github.com/nasa/fmdtools/blob/main/fmdtools_Individual_CLA.pdf>`_ , `Corporate CLA <https://github.com/nasa/fmdtools/blob/main/fmdtools_Corporate_CLA.pdf>`_).


Repository Structure
--------------------

.. image:: /docs-source/figures/uml/repo_organization.svg
   :width: 800
   
Getting started with development first requires some basic familiarity with the repo structure, shown above. As shown, the repo contains:

- ``/fmdtools``, which is where the toolkit sub-packages and modules are held,
- ``/tests``, which has tests of the modules,
- ``/examples``, which are different case study examples, 
- ``/docs-source``, which contains the sphinx documentation source files, and 
- ``/docs``, which contains documentation (built from source).

There are additionally a few scripts/config files with specific purposes to serve the development process:

- ``run_all_tests.py`` which is a script that runs tests defined in `/tests` and `/examples`,
- ``pyproject.toml`` which defines all python project and build configuration information,
- ``conf.py`` which defines sphinx documentation settings, and
- ``MAKE``, which is used to build the sphinx documentation.


Git Structure and Setup
-----------------------

.. image:: /docs-source/figures/uml/git_structure.svg
   :width: 800

Development of fmdtools uses a two-track development model, in which contributions are provided within NASA as well as by external collaborators. To support this, there are multiple repositories which must be managed in the development process, as shown above. Essentially, there is:

- An internal bitbucket, ``origin``, where NASA coordination, development, and continuous integration takes place,
- A public GitHub, ``public``, where collaboration with outside developers takes place (and where documentation is hosted), and 
- A PyPI repository, which contains stable versions of fmdtools which can be readily installed via ``pip``. This repository is automatically updated when a new version is released on GitHub.
- The fmdtools GitHub Pages site, which updates from the ``gh-pages`` branch.

The fmdtools team is responsible for coordinating the development between the internal and external git repositories. Managing multiple repositories can best be coordinated by:

- setting up the ``public`` and ``origin`` remotes on a single git repo on your machine
- using the ``dev`` branch and feature branches on ``origin`` for development and integration
- releasing to the ``main`` branch on ``origin`` and ``public``

To assist with this, the custom git alias below can be helpful::

	[alias]
		up = "!git merge dev main"
		tl = "!f() { git tag -s -a \"$1\" -m \"$2\"; }; f"
		pp = "!f() { git push public tag \"$1\"; }; f"
		po = "!f() { git push origin tag \"$1\"; }; f"
		release = "!f() { git checkout main && git up && git tl \"$1\" \"$2\" && git pp \"$1\" && git po \"$1\"; }; f"
		fb = "!f() { git fetch origin && git fetch public; }; f"
		mm = "!git merge main dev"
		sync-into-dev = "!f() { git checkout dev && git fb && git pull origin dev && mm; }; f"

Adding this block to your repository's git config file (e.g., ``.git/config``) adds custom git commands which can be used to simplify the release process. Specifically:

- ``git sync-into-dev`` will merge all main and dev branches (local and remote) into your local dev branch
- ``git release "vX.X.X" "Version X.X.X"`` will merge ``dev`` into ``main``, tag it with the given version, and upload it to ``public`` and ``origin``.

Note that the ``-s`` option above `signs` the tag, attributing it to your credentials. For this to work, you should `set up commit signing <https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits>`_ and configure git to sign commits by default.



Git Development Workflow
------------------------

.. image:: /docs-source/figures/uml/dev_process.svg
   :width: 800

To encourage code quality we follow the general process above to manage contributions:

1. Development begins on a ``/dev`` branch for a given version of fmdtools. This branch is used to test and integrate contributions from multiple sources.
2. The first step in making a contribution is then to create an issue, which describes the work to be performed and/or problem to be solved. 
3. This issue can then be taken on in an issue branch (or repo for external contributions) and fixed. 
4. When the contributor is done with a given fix, they can submit a pull request, which enables the review of the fix as a whole.
5. This fix is reviewed by a member of the fmdtools team, who may suggest changes. 
6. When the review is accepted, it is merged into the ``dev`` branch.
7. When all the issues for a given version are complete (this may also happen concurrently with development), tests and documentation are updated for the branch. If tests do not pass (or are obsolete), contributions may be made directly on the ``dev`` branch to fix it, or further issues may be generated based on the impact of the change.
8. When the software team deems the release process to be complete, the ``dev`` branch may be merged into the `main` branch. These branches are then used to create releases. 

The major exceptions to this process are:

- bug fixes, which, if minor, may occur on ``main``/``dev`` branches (or may be given their own branches off of ``main``),
- external contributions, which are managed via pull request off of ``main`` (or some external dev branch), and
- minor documentation changes.

Release Process
---------------

Releases are made to fmdtools to push new features and bugfixes to the open-source community as they are developed. Some important things to remember about the release process are:

* It's important to test prior to release to ensure (1) bugs aren't being released that could have been caught easily with a test (2) test results are accurate to the current version of the code and (3) the documentation stays up to date with the release. Currently, this is managed via Bamboo CI, which automatically builds releases on the /dev branch.

* Releases are made to the fmdtools GitHub repository using the ``Draft a new release`` button on the `Releases <https://github.com/nasa/fmdtools/releases>`_ page. Once this release is generated, GitHub Actions uploads it to the `fmdtools PyPi repo <https://pypi.org/project/fmdtools>`_ automatically.

To ensure all of the steps of the release process are performed, follow the :download:`Release Checklist <release_checklist.csv>`, see below:

.. tabularcolumns:: |p{1cm}|p{3cm}|p{1cm}|p{3cm}|

.. csv-table:: Release Checklist
   :file: release_checklist.csv
   :header-rows: 1
   :widths: 1 3 1 3

.. tabularcolumns:: |p{1cm}|p{3cm}|p{1cm}|p{3cm}|


Roles
-----

- team lead: 		coordinates all activities and has technical authority over project direction
- full developer: 	can make changes off of version and main branches and has full ability to perform the release process
- contributor: 		creates issues and develops off of issue branches


Documentation
-------------

Documentation is generated using Sphinx, which generates html from rst files. This is performed automatically on the Bamboo server. The process for generating documentation locally is to open powershell and run::
	
	cd path/to/fmdtools
	./make clean
	./make html

Note that building the docs page with sphinx requires the following dependencies, which should be installed beforehand::

	myst-nb
	sphinx_rtd_theme
	pandoc

Pandoc is an external program and thus should be installed from `its website <https://pandoc.org/installing.html>`_ or anaconda.

Style/Formatting
----------------

Generally, we try to follow PEP8 style conventions. To catch these errors, it is best to *turn on PEP8 style linting* in your IDE of choice.

Style conventions can additionally be followed/enforced automatically using the Black code formatter. See resources:

- stand-alone formatter: https://github.com/psf/black
- VSCode Extension: https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter
- Spyder workflow: https://stackoverflow.com/questions/55698077/how-to-use-code-formatter-black-with-spyder


Headers
-------

All source code files to be released under fmdtools should have the Apache-2.0 license applied to them. 

In modules and scripts, it is best practice to use the following format for the header::

	#!/usr/bin/env python
	# -*- coding: utf-8 -*-
	"""
	<One-line software description here>

	<Further module-level docstring information, if needed>

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

	<local/project imports>

	<fmdtools imports (examples, then tests, then modules)>

	<external imports>


For jupyter notebooks, the following block should be inserted at the end of the first markdown cell, after the title/description::

	```
	Copyright © 2024, United States Government, as represented by the Administrator of the National Aeronautics and Space Administration. All rights reserved.

	The “"Fault Model Design tools - fmdtools version 2"” software is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 

	Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
	```


Testing
-------

There are two major types of tests:

- quantitative tests, which are tests running ``run_all_tests.py``, and
- qualitative tests, which are the example notebooks


# -*- coding: utf-8 -*-
"""
Pytest configuration file.

Adds options to pytest:
    - testtype: str
        Defines type of tests to run. May be
            "full" if all tests
            "doctests" for just doctests
            "notebooks-fast"/"notebooks-slow"/"notebooks" for different lists of notebooks
            or any string for a custom test type
    - auto_build_reports : bool
        Whether to build a report given the testtype options. Default is False.
        If True, reports will go in the reports/testtype-pyver folder

Examples
--------
pytest --testtype=fast-notebooks  # should run all fast notebooks
pytest --testtype=doctests  # should run all doctest modules
pytest --testtype=doctests
# the below command should test one doctest file while 
# note that cov_report "html:auto" is required for a coverage report to be generated
# while
pytest --doctest-modules fmdtools/define/container/base.py --testtype=custom --auto_build_reports=True --cov-report "html:auto"
# typically, for testing, we just use this for fast tests:
pytest --testtype="doctest" -auto_build_reports=True --cov-report "html:auto"
# and this for comprehensive tests:
pytest --testtype="full" --auto_build_reports=True --cov-report "html:auto"


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
from matplotlib import pyplot as plt
import sys
# listing of modules with doctests
doctest_modules = ["fmdtools/define/container/base.py",
                   "fmdtools/define/container/state.py",
                   "fmdtools/define/container/parameter.py",
                   "fmdtools/define/container/mode.py",
                   "fmdtools/define/container/rand.py",
                   "fmdtools/define/container/time.py",
                   "fmdtools/define/object/base.py",
                   "fmdtools/define/object/timer.py",
                   "fmdtools/define/object/geom.py",
                   "fmdtools/define/object/coords.py",
                   "fmdtools/define/flow/base.py",
                   "fmdtools/define/architecture/function.py",
                   "fmdtools/define/architecture/action.py",
                   "fmdtools/define/architecture/component.py",
                   "fmdtools/define/architecture/geom.py",
                   "fmdtools/define/block/base.py",
                   "fmdtools/define/block/function.py",
                   "fmdtools/define/block/action.py",
                   "fmdtools/define/block/component.py",
                   "fmdtools/define/environment.py",
                   "fmdtools/sim/scenario.py",
                   "fmdtools/sim/sample.py",
                   "fmdtools/sim/search.py",
                   "fmdtools/sim/propagate.py",
                   "fmdtools/analyze/graph/style.py",
                   "fmdtools/analyze/graph/base.py",
                   "fmdtools/analyze/graph/model.py",
                   "fmdtools/analyze/result.py",
                   "fmdtools/analyze/history.py",
                   "fmdtools/analyze/phases.py",
                   "fmdtools/analyze/tabulate.py",
                   "fmdtools/analyze/common.py",
                   "examples/rover/rover_model.py",
                   "examples/rover/rover_model_human.py",
                   "examples/multirotor/drone_mdl_static.py",
                   "examples/multirotor/drone_mdl_dynamic.py",
                   "examples/multirotor/drone_mdl_hierarchical.py",
                   "examples/airspacelib/base/state.py",
                   "examples/airspacelib/base/arch/perceiveenvironment.py",
                   "examples/airspacelib/base/arch/controlflight.py",
                   "examples/airspacelib/base/arch/aviate.py"]

# list of fast-running notebooks:
fast_notebooks = ["examples/asg_demo/Action_Sequence_Graph.ipynb",
                  "examples/eps/EPS_Example_Notebook.ipynb",
                  "examples/multirotor/Demonstration.ipynb",
                  "examples/pump/Pump_Example_Notebook.ipynb",
                  "examples/pump/Tutorial_complete.ipynb",
                  "examples/rover/Model_Structure_Visualization_Tutorial.ipynb",
                  "examples/rover/FaultSample_Use-Cases.ipynb",
                  "examples/rover/Rover_Setup_Notebook.ipynb",
                  "examples/tank/Tank_Analysis.ipynb",
                  "examples/taxiway/Paper_Notebook.ipynb"
                  ]

# list of slow-running notebooks:
slow_notebooks = ["examples/multirotor/Multirotor_Optimization.ipynb",
                  "examples/pump/Optimization.ipynb",
                  "examples/rover/degradation_modelling/Degradation_Modelling_Notebook.ipynb",
                  "examples/rover/HFAC_Analyses/IDETC_Human_Paper_Analysis.ipynb",
                  "examples/pump/Parallelism_Tutorial.ipynb",
                  "examples/airspacelib/wildfireresponse/Wildfire_Demo.ipynb",
                  "examples/airspacelib/contingencymanagement/demo_notebook.ipynb",
                  "examples/airspacelib/contingencymanagement/proxthreat_notebook.ipynb"
                  ]

# for testing extremely slow notebooks that can't be run to completion :
too_slow_notebooks = ["examples/rover/HFAC_Analyses/HFAC_Analyses.ipynb",
                      "examples/pump/Stochastic_Modelling.ipynb", # timeout comes back as failed
                      "examples/multirotor/Urban_Drone_Demo.ipynb", # timeout comes back as failed
                      "examples/rover/optimization/Rover_Response_Optimization.ipynb",  # extremely slow notebook
                      "examples/rover/fault_sampling/Rover_Mode_Notebook.ipynb",  # extremely slow notebook
                      "examples/rover/optimization/Search_Comparison.ipynb",  # extremely slow
                      "examples/rover/ParameterSample_Use-Cases.ipynb", # timeout fails (over 300s)
                      "examples/tank/Tank_Optimization.ipynb",
                      "examples/airspacelib/wildfireresponse/paper_figures.ipynb"
                      ]


# tells pytest to ignore build files as well as overly slow notebooks
collect_ignore =  ["_build", "docs", "tmp", *too_slow_notebooks,
                   "examples/pump/Tutorial_unfilled.ipynb"]


def pytest_addoption(parser):
    """
    Add option for testtype, pyver, and auto_build_reports to pytest.

    See pytest API for details on usage.
    """
    parser.addoption("--testtype", action="store", default="doctests",
                     help="test type: full, doctests, or any other name",
                     type=str)
    parser.addoption("--auto_build_reports", action="store", default=False,
                     help="build_report: whether to build a report",
                     type=bool)


def pytest_configure(config):
    """
    Configure pytest given testtype and auto_build_reports options.

    If testtype is given, only runs the appropriate tests for that type. Also creates
    paths for html test and coverage reports given auto_build_reports option.
    """
    plt.rcParams['figure.max_open_warning'] = 50

    testtype = config.getoption("--testtype")
    pyver = "py"+str(sys.version_info.major)+str(sys.version_info.minor)
    reportdir = "./reports/"+testtype+"-"+pyver
    if config.getoption("--auto_build_reports") and 'cov_report' in config.option:
        if not config.option.cov_source:
            raise Exception("Coverage report will not build without --cov option.")
        config.option.cov_report['html'] = reportdir+"/coverage_html"
        config.option.cov_report['xml'] = reportdir+"/coverage/coverage.xml"

        config.option.htmlpath = reportdir+"/junit/report.html"
        config.option.xmlpath = reportdir+"/junit/junit.xml"
    if "full" not in testtype and "custom" not in testtype:
        config.args = []
    if "doctests" in testtype:
        config.doctestmodules=True
        if "custom" not in testtype:
            config.option.file_or_dir = str(doctest_modules)
            config.args.extend(doctest_modules)

    if "notebooks" in testtype:
        if "slow" in testtype:
            config.args.extend(slow_notebooks)
        elif "fast" in testtype:
            config.args.extend(fast_notebooks)
        else:
            config.args.extend(fast_notebooks+slow_notebooks)


def pytest_unconfigure(config):
    """Close open figures to keep computational resources low."""
    # this should close any open plots
    plt.close('all')


if __name__ == "__main__":
    # some test usages of pytest with local options
    import pytest
    # pytest.main(["--testtype=fast-notebooks"])
    pytest.main([*fast_notebooks,
                 "--testtype=custom",
                 "--auto_build_reports=True",
                 "--cov-report",
                 "html:auto"])

    pytest.main(["--doctest-modules", "fmdtools/define/container/base.py",
                 "--testtype=doctests-custom",
                 "--auto_build_reports=True",
                 "--cov-report",
                 "html:auto"])

    pytest.main(["--testtype=doctests"])



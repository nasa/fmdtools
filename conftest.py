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
import pytest
from pathlib import Path

# listing of modules with doctests
doctest_modules = ["src/fmdtools/define/container/base.py",
                   "src/fmdtools/define/container/state.py",
                   "src/fmdtools/define/container/parameter.py",
                   "src/fmdtools/define/container/mode.py",
                   "src/fmdtools/define/container/rand.py",
                   "src/fmdtools/define/container/time.py",
                   "src/fmdtools/define/object/base.py",
                   "src/fmdtools/define/object/timer.py",
                   "src/fmdtools/define/object/geom.py",
                   "src/fmdtools/define/object/coords.py",
                   "src/fmdtools/define/flow/base.py",
                   "src/fmdtools/define/architecture/function.py",
                   "src/fmdtools/define/architecture/action.py",
                   "src/fmdtools/define/architecture/component.py",
                   "src/fmdtools/define/architecture/geom.py",
                   "src/fmdtools/define/block/base.py",
                   "src/fmdtools/define/block/function.py",
                   "src/fmdtools/define/block/action.py",
                   "src/fmdtools/define/block/component.py",
                   "src/fmdtools/define/environment.py",
                   "src/fmdtools/sim/scenario.py",
                   "src/fmdtools/sim/sample.py",
                   "src/fmdtools/sim/search.py",
                   "src/fmdtools/sim/propagate.py",
                   "src/fmdtools/analyze/graph/style.py",
                   "src/fmdtools/analyze/graph/base.py",
                   "src/fmdtools/analyze/graph/model.py",
                   "src/fmdtools/analyze/result.py",
                   "src/fmdtools/analyze/history.py",
                   "src/fmdtools/analyze/phases.py",
                   "src/fmdtools/analyze/tabulate.py",
                   "src/fmdtools/analyze/common.py",
                   "examples/navigating_rover/model_main.py",
                   "examples/navigating_rover/model_human.py",
                   "examples/multirotor_drone/model_static.py",
                   "examples/multirotor_drone/model_dynamic.py",
                   "examples/multirotor_drone/model_hierarchical.py",
                   "examples/airspacelib/base/state.py",
                   "examples/airspacelib/base/arch/perceiveenvironment.py",
                   "examples/airspacelib/base/arch/controlflight.py",
                   "examples/airspacelib/base/arch/aviate.py"]

# list of fast-running notebooks:
fast_notebooks = ["examples/human_hazard_mitigation/tutorial_actionarchitecture.ipynb",
                  "examples/electric_power_system/demo_static_models.ipynb",
                  "examples/state_communication/tutorial_MultiFlow_and_CommsFlow.ipynb"
                  "examples/multirotor_drone/demo_overview.ipynb",
                  "examples/multirotor_drone/paper_ijphm_fmdtools.ipynb",
                  "examples/multirotor_drone/tutorial_fmdtools_basics.ipynb",
                  "examples/water_pump/demo_fault_analysis.ipynb",
                  "examples/water_pump/tutorial_fmdtools_basics.ipynb",
                  "examples/navigating_rover/demo_overview.ipynb",
                  "examples/navigating_rover/tutorial_model_structure_visualization.ipynb",
                  "examples/navigating_rover/tutorial_FaultSample.ipynb",
                  "examples/navigating_rover/navigating_demo_rover_model.ipynb",
                  "examples/cooling_tank/demo_tank_model.ipynb",
                  "examples/airport_taxiway/paper_jcise_dsa.ipynb"
                  ]

# list of slow-running notebooks:
slow_notebooks = ["examples/multirotor_drone/multirotor_drone_Optimization.ipynb",
                  "examples/water_pump/tutorial_optimization.ipynb",
                  "examples/navigating_rover/demo_degradation.ipynb",
                  "examples/navigating_rover/paper_idetc_human.ipynb",
                  "examples/water_pump/tutorial_parallelism.ipynb",
                  "examples/airspacelib/wildfire_response/demo_wildfire.ipynb",
                  "examples/airspacelib/contingency_management/demo_contingency.ipynb",
                  "examples/airspacelib/contingency_management/demo_proxthreat.ipynb"
                  ]

# for testing extremely slow notebooks that can't be run to completion :
too_slow_notebooks = ["examples/navigating_rover/paper_ifac_human.ipynb",
                      "examples/water_pump/tutorial_stochastic_behavior.ipynb", # timeout comes back as failed
                      "examples/multirotor_drone/demo_urban_flight.ipynb", # timeout comes back as failed
                      "examples/navigating_rover/demo_response_optimization.ipynb",  # extremely slow notebook
                      "examples/navigating_rover/paper_jmd_synthetic_modes.ipynb",  # extremely slow notebook
                      "examples/navigating_rover/paper_aiaa_coevolution.ipynb",  # extremely slow
                      "examples/navigating_rover/tutorial_ParameterSample.ipynb", # timeout fails (over 300s)
                      "examples/cooling_tank/paper_jmd_optimization.ipynb",
                      "examples/airspacelib/wildfire_response/paper_aiaa_optimal_location.ipynb",
                      "conf.py"]

# tells pytest to ignore build files as well as overly slow notebooks
collect_ignore =  ["_build", "docs", "tmp", "conf", *too_slow_notebooks,]


def pytest_addoption(parser):
    """
    Add option for testtype, pyver, and auto_build_reports to pytest.

    See pytest API for details on usage.
    """
    parser.addoption("--skiplist", action="store_true",
                 default=too_slow_notebooks, help="skip listed tests")
    parser.addoption("--testtype", action="store", default="doctests",
                     help="test type: full, doctests, or any other name",
                     type=str)
    parser.addoption("--auto_build_reports", action="store", default=False,
                     help="build_report: whether to build a report",
                     type=bool)


def pytest_collection_modifyitems(config, items):
    """Skip listed (too slow notebooks) by default."""
    tests_to_skip = config.getoption("--skiplist")
    if not tests_to_skip:
        # --skiplist not given in cli, therefore move on
        return
    skip_listed = pytest.mark.skip(reason="included in --skiplist")
    for item in items:
        for testpath in tests_to_skip:
            if Path(testpath).samefile(item.path):
                item.add_marker(skip_listed)


def pytest_configure(config):
    """
    Configure pytest given testtype and auto_build_reports options.

    If testtype is given, only runs the appropriate tests for that type. Also creates
    paths for html test and coverage reports given auto_build_reports option.
    """
    plt.rcParams['figure.max_open_warning'] = 50

    testtype = config.getoption("--testtype")
    pyver = "py"+str(sys.version_info.major)+str(sys.version_info.minor)
    reportdir = ".tests/reports/"+testtype+"-"+pyver
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

    pytest.main(["--doctest-modules", "src/fmdtools/define/container/base.py",
                 "--testtype=doctests-custom",
                 "--auto_build_reports=True",
                 "--cov-report",
                 "html:auto"])

    pytest.main(["--testtype=doctests"])



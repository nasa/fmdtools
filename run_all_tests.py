#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing script for fmdtools.

Requires pytest, nbmake, pytest-html, pytest-cov.


Options
--------
--doctests: Bool
    Whether or not to run/collect doctests
--notebooks: Bool
    Whether or not to run/collect notebooks
--testtype : str
    Set test configuration to run. Could be "full" (collect all tests),
    "full-notebooks" (just notebooks), "fast-notebooks" (just fast notebooks),
    "slow-notebooks" (just slow notebooks), or "doctests" (just doctests).
    Default is "doctests" to enable fast testing.
--cov : bool
    Whether or not to produce coverage report. Default is True.
--report : bool
    Whether or not to produce test report. Default is True.

Examples
--------
# by default, run all tests (except slow notebooks):
# python -m run_all_tests
# run a faster configuration with no notebooks
# python -m run_all_tests --notebooks False
# run just module-level doctests with no report
# python -m run_all_tests --testtype "doctests"
"""
import pytest
import argparse

# enables large numbers of tests with plots to be run:
from matplotlib import pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50


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
                   "fmdtools/analyze/graph/style.py",
                   "fmdtools/analyze/graph/base.py",
                   "fmdtools/analyze/result.py",
                   "fmdtools/analyze/history.py",
                   "fmdtools/analyze/phases.py",
                   "fmdtools/analyze/tabulate.py",
                   "fmdtools/analyze/common.py",
                   "examples/rover/rover_model.py",
                   "examples/rover/rover_model_human.py",
                   "examples/multirotor/drone_mdl_static.py",
                   "examples/multirotor/drone_mdl_dynamic.py",
                   "examples/multirotor/drone_mdl_hierarchical.py"]


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
                  "examples/pump/Parallelism_Tutorial.ipynb"]

# for testing extremely slow notebooks that can't be run to completion :
too_slow_notebooks = ["examples/rover/HFAC_Analyses/HFAC_Analyses.ipynb",
                      "examples/pump/Stochastic_Modelling.ipynb", # timeout comes back as failed
                      "examples/multirotor/Urban_Drone_Demo.ipynb", # timeout comes back as failed
                      "examples/rover/optimization/Rover_Response_Optimization.ipynb",  # extremely slow notebook
                      "examples/rover/fault_sampling/Rover_Mode_Notebook.ipynb",  # extremely slow notebook
                      "examples/rover/optimization/Search_Comparison.ipynb",  # extremely slow
                      "examples/rover/ParameterSample_Use-Cases.ipynb", # timeout fails (over 300s)
                      "examples/tank/Tank_Optimization.ipynb"
                      ]


# we ignore the following notebooks for various reasons:
# while not included in the testing approach, they should be verified periodically
ignore_notebooks = [*too_slow_notebooks,
                    "examples/pump/Tutorial_unfilled.ipynb",  # intended to be blank
                    "_build",
                    "docs",
                    "tmp"]


def main(doctests=True, notebooks=True, testlist=[], testtype="full",
         ignore=ignore_notebooks, cov=True, report=True, pyver="py311"):
    pytestargs = ["--cache-clear"]

    # if doctests, add doctest_modules to set of tests
    # options for notebooks to test (provided testlist or set testlists)
    if doctests:
        pytestargs.extend(["--doctest-modules"])
    if testlist:
        testtype = "custom"
        pytestargs.extend(testlist)
    elif testtype == "full-notebooks":
        pytestargs.extend(fast_notebooks + slow_notebooks)
    elif testtype == "doctests":
        pytestargs.extend(doctest_modules)
    elif testtype == "fast-notebooks":
        pytestargs.extend(fast_notebooks)
    elif testtype == "slow-notebooks":
        pytestargs.extend(slow_notebooks)
    elif testtype == "too-slow-notebooks":
        pytestargs.extend(too_slow_notebooks)
    elif testtype != "full":
        raise Exception("Invalid testtype: "+testtype)

    reportdir = "./reports/"+testtype+"-"+pyver
    # adds coverage report
    if cov:
        pytestargs.extend(["--cov-report",
                           "html:"+reportdir+"/coverage_html",
                           "--cov-report",
                           "xml:"+reportdir+"/coverage/coverage.xml",
                           "--cov",])

    # adds html report
    if report:
        pytestargs.extend(["--html="+reportdir+"/junit/report.html",
                           "--junitxml="+reportdir+"/junit/junit.xml",
                           "--overwrite"])
    # adds notebooks
    if notebooks:
        pytestargs.extend(["--nbmake"])

    pytestargs.extend(["--ignore="+f for f in ignore])
    pytestargs.extend(["--continue-on-collection-errors"])
    try:
        pytest.main(pytestargs)
    except UnicodeEncodeError as e:
        # error handling for common (difficult-to-debug) report generation error
        raise Exception("UnicodeEncodeError resulting from non-standard characters" +
                        "in the console. Make sure all propagate methods use the" +
                        "showprogress=False option") from e

    # this should close any open plots
    plt.close('all')


if __name__ == "__main__":

    # retcode = pytest.main(["--cov-report",
    #                     "html:./reports/coverage",
    #                     "--cov-report",
    #                     "xml:./reports/coverage/coverage.xml",
    #                     "--cov",
    #                     "--html=./reports/junit/report.html",
    #                     "--junitxml=./reports/junit/junit.xml",
    #                     "--overwrite",
    #                     "--doctest-modules",
    #                     "--nbmake",
    #                     *["--ignore="+notebook for notebook in ignore_notebooks],
    #                     "--continue-on-collection-errors"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--doctests", default=True, required=False)
    parser.add_argument("--notebooks", default=True, required=False)
    parser.add_argument("--testtype", default="doctests", required=False)
    parser.add_argument("--cov", default=True, required=False)
    parser.add_argument("--report", default=True, required=False)
    parser.add_argument("--pyver", default="py311", required=False)
    parsed_args = parser.parse_args()
    kwargs = {k: v for k, v in vars(parsed_args).items() if v is not None}
    # main(**kwargs)
    main(notebooks=False, testtype="doctests", cov=False, report=False)
    # main(testtype="custom", testlist=["examples/tank/Tank_Analysis.ipynb", "examples/tank/test_tank.py"])
    # main(testtype="custom", testlist=["examples/pump/test_pump.py"])
    # main(testtype="custom", testlist=["examples/rover/test_rover.py"])
    # after creating test report, update the badge using this in powershell:
    # !Powershell.exe -Command "genbadge tests"
    # for coverage report, remove .gitignore from /reports/coverage
    # !Powershell.exe -Command "genbadge coverage"

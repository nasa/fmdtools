# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:12:39 2023

@author: dhulse
"""
import pytest
# import sys

# NOTE: If report won't generate with error:
# "UnicodeEncodeError: 'charmap' codec can't encode characters in position..."
# make sure all tests pass show_progress=False to propagate

if __name__=="__main__":
    # requires pytest, nbmake, pytest-html
    
    # for testing modules with doctests
    doctest_modules = ["fmdtools/define/state.py", "fmdtools/define/parameter.py"]
    redcode = pytest.main(["--doctest-modules", *doctest_modules])
    
    # retcode = pytest.main(["--html=./reports/junit/report.html",
    #                        "--self-contained-html",
    #                        "--junitxml=./reports/junit/junit.xml",
    #                        "--doctest-modules",
    #                        "--continue-on-collection-errors"])

    fast_notebooks = ["examples/asg_demo/Action_Sequence_Graph.ipynb",
                      "examples/eps/EPS_Example_Notebook.ipynb", 
                      "examples/pump/Pump_Example_Notebook.ipynb",
                      "examples/pump/Stochastic_Modelling.ipynb",
                      "examples/rover/Approach_Use-Cases.ipynb",
                      "examples/rover/Model_Structure_Visualization_Tutorial.ipynb",
                      "examples/rover/Nominal_Approach_Use-Cases.ipynb",
                      "examples/rover/Rover_Setup_Notebook.ipynb",
                      "examples/tank/Tank_Analysis.ipynb"
                      ]

    # for testing notebooks during development:
    # retcode = pytest.main(["--nbmake", *fast_notebooks])

    slow_notebooks = ["examples/multirotor/Demonstration.ipynb",
                      "examples/multirotor/Multirotor_Optimization.ipynb",
                      "examples/pump/AST_Sampling.ipynb",
                      "examples/pump/Optimization.ipynb",
                      "examples/pump/Parallelism_Tutorial.ipynb",
                      "examples/pump/IDETC Results/IDETC_Figures.ipynb",
                      "examples/rover/degradation_modelling/Degradation Modelling Notebook.ipynb",
                      "examples/rover/fault_sampling/Rover Mode Notebook.ipynb",
                      "examples/rover/HFAC_Analyses/HFAC Analyses.ipynb",
                      "examples/rover/HFAC_Analyses/IDETC_Human_Paper_Analysis.ipynb",
                      "examples/rover/optimization/Rover Response Optimization.ipynb",
                      "examples/rover/optimization/Search Comparison.ipynb",
                      "examples/tank/Tank Optimization.ipynb"]

    # for testing longer-running notebooks
    # retcode = pytest.main(["--nbmake", *slow_notebooks])

    # for testing all unittests
    # retcode = pytest.main()

    # for creating comprehensive test report:
# =============================================================================
#     retcode = pytest.main(["--html=./reports/junit/report.html",
#                            "--junitxml=./reports/junit/junit.xml",
#                            "--nbmake",
#                            "--overwrite",
#                            "--doctest-modules",
#                            "--continue-on-collection-errors"])
# =============================================================================
    
    # after creating test report, update the badge using this in powershell:
    # !Powershell.exe -Command "genbadge tests"
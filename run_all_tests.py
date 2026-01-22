#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing script for fmdtools internal build environment.

Requires pytest, nbmake, pytest-html, pytest-cov.
"""
import pytest
import argparse
import sys
import os


def main(testtype="doctests", auto_build_reports=False):
    rep_args=[]
    if auto_build_reports:
        rep_args = ["--cov-report", "html:auto"]

    print("os cwd: "+os.getcwd())
    print("sys paths: "+"\n ".join(sys.path))

    try:
        pytest.main(["-v")
                     #"--testtype="+testtype,
                     #"--auto_build_reports="+str(auto_build_reports), *rep_args])
    except:
        print("os cwd: "+os.getcwd())
        print("sys paths: "+"\n ".join(sys.path))
        import conftest


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--testtype", default="doctests", required=False)
    parser.add_argument("--auto_build_reports", default=True, required=False)
    parsed_args = parser.parse_args()
    kwargs = {k: v for k, v in vars(parsed_args).items() if v is not None}
    main(**kwargs)


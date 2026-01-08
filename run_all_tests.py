#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Testing script for fmdtools internal build environment.

Requires pytest, nbmake, pytest-html, pytest-cov.
"""
import conftest
import pytest
import argparse


def main(testtype="doctests", auto_build_reports=False):
    rep_args=[]
    if auto_build_reports:
        rep_args = ["--cov-report", "html:auto"]
    pytest.main(["--testtype="+testtype,
                 "--auto_build_reports="+str(auto_build_reports), *rep_args])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--testtype", default="doctests", required=False)
    parser.add_argument("--auto_build_reports", default=True, required=False)
    parsed_args = parser.parse_args()
    kwargs = {k: v for k, v in vars(parsed_args).items() if v is not None}
    main(**kwargs)


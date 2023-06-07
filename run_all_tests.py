# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:12:39 2023

@author: dhulse
"""
import pytest

if __name__=="__main__":
    # requires pytest, nbmake, pytest-html
    retcode = pytest.main(["--html=pytest_report.html", "--nbmake"])
    
    #retcode = pytest.main(["--html=pytest_report.html", "--continue-on-collection-errors"])
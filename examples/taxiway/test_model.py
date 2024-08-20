#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests of integrated taxiway model.

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

from model import taxiway_model
from common import plot_course, plot_tstep

from fmdtools.sim import propagate as prop
from fmdtools.analyze import phases
from fmdtools.analyze.common import suite_for_plots

import unittest
from matplotlib import pyplot as plt



class ModelTests(unittest.TestCase):
    def setUp(self):
        self.mdl = taxiway_model()

    def test_scen(self):
        """Tests that every asset has cycled after 120 timesteps. Should not crash."""
        for t in range(120):
            self.mdl.propagate(t)
        uncycled = [f for f, fxn in self.mdl.fxns.items()
                    if f != "atc" and not fxn.s.cycled]
        self.assertFalse(any(uncycled))
        num_crash = len([f for f, fxn in self.mdl.fxns.items()
                         if f != "atc" and fxn.m.has_fault("crash")])
        self.assertEqual(num_crash, 0)

    def test_default_plots(self):
        """Verify that the assets cycle through every mode/circulate over the taxiway"""
        res, hist = prop.nominal(self.mdl)
        for fxn in self.mdl.fxns:
            if fxn != "atc":
                fig, ax = plot_course(self.mdl, hist, fxn, title=fxn)
                plt.close(fig)
        phasemaps = phases.from_hist(hist)
        phases.phaseplot(phasemaps)

    def test_atc_lost_ground_perception(self):
        """Verify that the atc losing ground perception results in aircraft being stuck
        in the air."""
        res, hist = prop.one_fault(self.mdl, "atc", "lost_ground_perception", time=10)
        # hist.faulty.atc
        # hist.faulty.atc.area_allocation
        self.assertTrue(any(hist.faulty.fxns.atc.m.faults.lost_ground_perception))
        self.assertTrue(res.endclass.perc_cycled <= 0.5)

    def test_atc_lost_ground_perception_plot(self):
        """Plot for atc losing ground perception."""
        res, hist = prop.one_fault(self.mdl, "atc", "lost_ground_perception", time=10)
        plot_tstep(self.mdl, hist.faulty, 110, title="Aircraft backed up in air")

    def test_atc_wrong_land_command(self):
        """Test that wrong landing command by itself does not result in crashes."""
        res, hist = prop.one_fault(self.mdl, "atc", "wrong_land_command", time=5)
        self.assertTrue(res.endclass.num_crashed == 0)

    def test_atc_wrong_land_command_lost_sight(self):
        """Tests that wrong landing commands result in crashes"""
        seq = {1: {"atc": "wrong_land_command"}, 2: {"h2": "lost_sight"}}
        res, hist = prop.sequence(self.mdl, faultseq=seq)
        self.assertTrue(res.endclass.num_crashed >= 1)

    def test_atc_wrong_land_command_sight_plot(self):
        seq = {1: {"atc": "wrong_land_command"}, 2: {"h2": "lost_sight"}}
        res, hist = prop.sequence(self.mdl, faultseq=seq)
        phasemaps = phases.from_hist(hist.faulty)
        phases.phaseplot(phasemaps, title="helicopters should not cycle")


if __name__ == "__main__":

    runner = unittest.TextTestRunner()
    runner.run(suite_for_plots(ModelTests, plottests=False))
    runner.run(suite_for_plots(ModelTests, plottests=True))

    # unittest.main()

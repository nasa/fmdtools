# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:27:19 2023

@author: dhulse
"""
from model import taxiway_model
from common import plot_course, plot_tstep

from fmdtools.sim import propagate as prop
import fmdtools.analyze as an
import unittest
from fmdtools.analyze.common import suite_for_plots


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
        phasemaps = an.phases.from_hist(hist)
        an.phases.phaseplot(phasemaps)

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
        phasemaps = an.phases.from_hist(hist.faulty)
        an.phases.phaseplot(phasemaps, title="helicopters should not cycle")


if __name__ == "__main__":

    runner = unittest.TextTestRunner()
    runner.run(suite_for_plots(ModelTests, plottests=False))
    runner.run(suite_for_plots(ModelTests, plottests=True))

    # unittest.main()

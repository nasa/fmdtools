"""Regression tests for phase-map timesteps inferred after simulation."""

import unittest

import numpy as np

from fmdtools.analyze.phases import from_hist
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.define.container.mode import Fault
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample


class TestGeneratedHistoryPhaseTimestep(unittest.TestCase):
    """Keep history-derived fault opportunities on the configured model grid."""

    def test_generated_maps_preserve_grid_exposure_and_inputs(self):
        for dt in (0.25, 0.5, 1.0, 2.0):
            with self.subTest(dt=dt):
                model = ExFxnArch(sp={"dt": dt, "end_time": 4 * dt, "use_local": False})
                sim = propagate.Simulation(mdl=model)
                _, hist = sim()
                before = hist.copy()
                app = sim.gen_sampleapproach(get_phasemap=True)
                for name in ("ex_fxn", "ex_fxn2"):
                    mapping = app.phasemaps[name]
                    self.assertEqual(mapping.dt, dt)
                    np.testing.assert_allclose(
                        mapping.get_phase_times("standby"),
                        np.arange(5) * dt,
                        rtol=0,
                        atol=1e-12,
                    )
                    self.assertAlmostEqual(
                        mapping.calc_modephase_time("standby"), 5 * dt
                    )
                    fault = Fault(prob=0.2, units="hr")
                    self.assertAlmostEqual(
                        fault.calc_rate(0.0, mapping, sim_units="hr"), 0.2 * 5 * dt
                    )
                for key, value in before.flatten().items():
                    np.testing.assert_array_equal(hist.flatten()[key], value)
                self.assertEqual(model.sp.dt, dt)

    def test_call_returned_approach_drives_fractional_fault_samples(self):
        dt = 0.5
        sim = propagate.Simulation(
            mdl=ExFxnArch(sp={"dt": dt, "end_time": 2.0, "use_local": False})
        )
        _, _, app = sim(gen_samp=True, get_phasemap=True)
        domain = FaultDomain(sim.mdl)
        domain.add_fault("ex_fxn", "low")
        sample = FaultSample(domain, phasemap=app.phasemaps["ex_fxn"])
        sample.add_fault_phases("standby", method="all")
        self.assertEqual(sorted(sample.get_times()), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(sample.num_scenarios(), 5)
        np.testing.assert_allclose(
            [s.rate for s in sample.scenarios()], np.full(5, 0.2)
        )
        self.assertAlmostEqual(sum(s.rate for s in sample.scenarios()), 1.0)

    def test_history_changes_of_mode_keep_fractional_exposure(self):
        sim = propagate.Simulation(
            mdl=ExFxnArch(sp={"dt": 0.5, "end_time": 2.0, "use_local": False})
        )
        sim()
        # An existing history can include repeated operating modes.
        flat = sim.history.flatten()
        flat["fxns.ex_fxn.m.mode"] = np.array(
            ["standby", "on", "on", "standby", "standby"]
        )
        sim.history = flat
        app = sim.gen_sampleapproach(get_phasemap=True)
        mapping = app.phasemaps["ex_fxn"]
        self.assertEqual(mapping.dt, 0.5)
        np.testing.assert_allclose(mapping.get_phase_times("on"), [0.5, 1.0])
        self.assertAlmostEqual(mapping.calc_modephase_time("on"), 1.0)
        np.testing.assert_allclose(mapping.get_phase_times("standby"), [0.0, 1.5, 2.0])
        self.assertAlmostEqual(mapping.calc_modephase_time("standby"), 1.5)

    def test_opt_out_and_standalone_extractor_defaults_are_unchanged(self):
        sim = propagate.Simulation(
            mdl=ExFxnArch(sp={"dt": 0.5, "end_time": 2.0, "use_local": False})
        )
        sim()
        self.assertEqual(set(sim.gen_sampleapproach().phasemaps), {"mdl"})
        extracted = from_hist(sim.history)
        self.assertTrue(all(mapping.dt == 1.0 for mapping in extracted.values()))


if __name__ == "__main__":
    unittest.main()

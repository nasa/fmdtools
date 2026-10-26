Satellite risk and FMEA
=======================

This workflow connects the existing satellite's seeded disturbances and explicit
faults to ``FaultSample``, ``propagate.fault_sample``, ``Result.get_expected``
and ``FMEA``. It leaves both satellite models and the core library unchanged.
The executable companion is ``demo_risk_analysis.ipynb``.

Metric and scope
----------------

The default delivery target is two packets by minute six. For each simulation,
``shortfall = max(required_packets - downlinked, 0)`` counts missing packets.
``delivery_failed`` is one when shortfall is positive, otherwise zero.
``delta_shortfall`` subtracts the same seed's no-fault shortfall and may be
negative. These are count-based outcomes, without a monetary cost model.

A biased payload can select extra packets and sometimes improve this delivery
count. That does not establish better scientific observations: the model has
no truth labels, packet identities or classifier-accuracy metric. It also has
no physical spacecraft dynamics or calibrated reliability inputs.

Four single-fault modes are analysed: capture failure, capture bias, processing
failure and downlink failure. Their inherited, illustrative per-run rates are
0.01, 0.02, 0.01 and 0.01. The default injection times are minutes 1 and 3 with
conditional time weights 0.25 and 0.75 for each mode. These timing assumptions
are explicit modelling choices, not an estimated onset distribution.

Weighting convention
--------------------

For fault mode ``m``, time ``t`` and disturbance seed ``s``, let ``L(m,t,s)``
be shortfall, ``p(m)`` the declared fault rate, ``q(t)`` the conditional time
weight and ``N`` the number of seeds. The FMEA columns mean:

* ``average_shortfall`` = sum over t of q(t) * mean over s of L(m,t,s).
  This is severity conditional on the fault, in packets.
* ``expected_shortfall`` = p(m) * average_shortfall. This is a fault-weighted
  contribution, in packets per synthetic run. The ``expected_delivery_failed``
  column applies the same weights to the event indicator.
* ``sum_scenario_rate`` = p(m), because q(t) sums to one. Repeating the
  experiment across more seeds must not multiply this mass.

The corresponding ``average_``/``expected_delta_shortfall`` columns use the
paired change from the no-fault run. Conditional averages use q(t), while
expected columns use p(m)*q(t); rates are not applied twice.

Each distinct seed has weight 1/N. Samples already come from the configured
Gaussian/Bernoulli distributions, so their latent densities are never used as
additional weights. A Gaussian density is not a discrete event probability.
No-fault disturbance losses are retained separately as ``nominal`` results and
excluded from the FMEA rows. Here nominal means no injected persistent fault;
random disturbances can still prevent delivery.

The fault catalogue is not an exhaustive, mutually exclusive mission-state
model. Its weighted sum excludes joint faults and no-fault background losses.
It must not be labelled total mission failure probability or total mission
risk. In particular, this example does not invent a nominal probability of
``1 - sum(fault rates)`` or add background and fault totals together.

Deterministic reference
-----------------------

Disable randomness for an independently checkable calculation::

    from fmdtools_examples.planetary_observation.script_risk_analysis import run_risk_analysis

    reference = run_risk_analysis()
    table = reference.fmea.as_table(sort_by="expected_shortfall")
    print(table)
    assert abs(sum(reference.fmea["expected_shortfall"].values()) - 0.045) < 1e-12

Without an injected fault, both selected packets are delivered. Capture and
processing failures at minute 1 cause a two-packet shortfall; failures at
minute 3 cause a one-packet shortfall. Their conditional severity is therefore
``0.25*2 + 0.75*1 = 1.25`` packets. Downlink failure at either time causes a
two-packet shortfall. Positive payload bias does not reduce the delivery count.

.. list-table:: Deterministic reference
   :header-rows: 1

   * - Fault
     - Declared rate
     - Mean shortfall
     - Weighted shortfall
   * - Capture failure
     - 0.01
     - 1.25
     - 0.0125
   * - Capture bias
     - 0.02
     - 0.00
     - 0.0000
   * - Processing failure
     - 0.01
     - 1.25
     - 0.0125
   * - Downlink failure
     - 0.01
     - 2.00
     - 0.0200

The catalogue's weighted shortfall is **0.045 packets** and its weighted
failure-indicator contribution is **0.03**. These references test the software
and the stated assumptions, rather than estimating real spacecraft risk.

Seeded disturbance analysis
---------------------------

Run the same catalogue for each seed, keeping no-fault and faulty runs paired::

    analysis = run_risk_analysis(seeds=range(32), stochastic=True)
    print(analysis.fmea.as_table(sort_by="expected_shortfall"))
    print("Background mean shortfall:", analysis.mean_result["nominal.tend.classify.shortfall"])
    print("Background failure fraction:", analysis.mean_result["nominal.tend.classify.delivery_failed"])

``results`` and ``histories`` contain every realization under ``seed_0``, etc.
``mean_result`` retains the average of each terminal metric for each scenario.
The complete realization histories and scenario rates remain inspectable::

    first_results = analysis.results.get("seed_0")
    first_histories = analysis.histories.get("seed_0")
    scenario = analysis.sample.scenarios()[0]
    print(scenario.name, scenario.time, scenario.rate)
    print(first_results.get(scenario.name))
    print(first_histories.get(scenario.name))

The notebook shows a distribution over complete per-seed catalogue totals and
an estimated Monte Carlo standard error of their mean. Fault scenarios sharing
a seed are paired, so they are not treated as independent uncertainty samples.
This sampling error does not include uncertainty in rates, model structure or
the chosen input distributions. Thirty-two seeds are a small runnable example,
not evidence of convergence or a guarantee of capturing rare events.

Reproducibility requires the same seed list, model parameters, time weights,
mission target and compatible dependency versions. ``staged=False`` runs the
same workflow without state reuse. Tests compare both execution paths.

Exports and tests
-----------------

The returned FMEA supports the existing ``as_table`` and ``as_plot`` methods.
For an explicit local export, use a dedicated output directory::

    from pathlib import Path

    output = Path("outputs_script_risk_analysis")
    output.mkdir(exist_ok=True)
    analysis.fmea.as_table().to_csv(output / "fmea.csv")
    analysis.results.save(str(output / "results.npz"))
    analysis.histories.save(str(output / "histories.npz"))

Also retain the inputs above when archiving results; raw outputs alone do not
specify the experiment. Existing files are protected by the result save API.
The script entry point prints the deterministic table without writing files::

    python -m fmdtools_examples.planetary_observation.script_risk_analysis
    python -m pytest --noconftest -o addopts= --import-mode=importlib examples/planetary_observation/test_risk_analysis.py

Tests check analytical scenario rates and severities, all four equally likely
contact-loss outcomes in a controlled reference, equal seed weighting, paired
histories, signed differences, staged/unstaged equivalence, zero weights, late
faults, invalid inputs, sorted/CSV tables and input ownership. Example-specific
tests remain beside the model. ML integration remains a separate increment.

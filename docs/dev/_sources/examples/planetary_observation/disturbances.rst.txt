Random disturbances and injected faults
=======================================

``model_disturbances.py`` extends the deterministic satellite without changing
``model_main.py`` or the library. The schedule, one-minute timestep, threshold
selector and count-only buffer retain their original meaning. These are
synthetic software experiments, not a calibrated spacecraft model.

Random inputs
-------------

``DisturbanceParam`` extends the mission parameters with:

* ``noise_std=0.05``: standard deviation of zero-mean Gaussian score noise.
  A captured score is clipped to [0, 1] after adding noise and any fault bias.
  Zero scale bypasses the Gaussian draw, including its density calculation.
* ``obscuration_prob=0.1``: probability that a scheduled observation is
  temporarily unavailable. This is a Bernoulli draw per observation.
* ``contact_loss_prob=0.1``: probability that a scheduled contact is unavailable.
  Queued packets remain in the buffer; there is no retransmission within a step.
* ``bias=0.4``: additive score offset while the payload's ``biased`` fault is
  active. This is a fault parameter, not an additional random draw.

The schedule function owns a native ``Rand`` stream. At each observation it
samples noise, then obscuration; at each contact it samples link loss. The
contact draw follows the observation draws when both events coincide. Idle
steps and the initial condition consume no draws. Draws depend on the schedule
and parameters, never on an injected fault's state. This keeps exogenous inputs
paired between nominal and faulty runs with the same seed and configuration.
It also avoids separate functions accidentally using identical seeded streams.
The model assumes independent disturbance draws, not persistent cloud cover or
correlated environmental outages.

Enable stochastic behavior explicitly::

    from fmdtools.sim import propagate
    from fmdtools_examples.planetary_observation.model_disturbances import (
        DisturbedObservationSatellite,
    )

    model = DisturbedObservationSatellite(
        r={"seed": 0},
        sp={"run_stochastic": True, "track_pdf": True},
    )
    result, history = propagate.nominal(model)
    print(result)
    print(history["fxns.schedule.r.s.noise"])

Seed zero is valid. Replay is tested within the installed dependency versions;
this example makes no cross-version bitwise reproducibility guarantee. With
``run_stochastic=False``, all temporary disturbances are disabled, even when
a configured probability is one. With all three disturbance parameters zero,
stochastic execution also reproduces the deterministic scaffold.

With ``track_pdf=True``, ``fxns.schedule.r.probdens`` records the product of
masses/densities of the *latent inputs drawn at that step*. It is one for an
empty draw. A Gaussian density can exceed one; this quantity is neither the
probability of the clipped observation nor a mission-failure probability.
No random disturbance probabilities are multiplied into ``Fault.prob``.

Persistent faults
-----------------

Faults are activated explicitly through the normal propagation/sampling API.
A ``Mode`` probability does not spontaneously activate the mode during a run.
All probabilities below are independent, illustrative per-run inputs in
``units='sim'``; they are not measured rates or validated reliability estimates.

* ``capture.failed`` (0.01): prevents capture from its injection time onward.
* ``capture.biased`` (0.02): adds ``bias`` before clipping a valid observation.
  If both payload faults are active, ``failed`` takes precedence.
* ``process.failed`` (0.01): discards new captured packets before selection.
  It does not erase previously queued packets.
* ``downlink.failed`` (0.01): blocks all subsequent transmission. The buffer
  retains packets, so overflow can occur if later observations continue.

For a hand-checkable fault run, leave randomness disabled::

    model = DisturbedObservationSatellite()
    result, history = propagate.one_fault(
        model, "capture", "failed", time=3.0, staged=True,
    )
    print(result)

The faulty run acquires the first two observations, misses the last two,
selects one packet and delivers it. Failure is persistent after injection;
this increment does not model recovery or repair. Fault injection at time zero
prevents all subsequent acquisitions, while injection after the final
observation cannot retroactively remove previously captured data.

Accounting and scope
--------------------

The inherited raw counts are extended with ``missed`` (scheduled acquisitions
blocked by obscuration or payload failure) and ``unprocessed`` (captures
rejected during processing failure). The following hold at every step::

    opportunities = acquired + missed
    acquired = processed + unprocessed
    selected = downlinked + queued + dropped

``dropped`` continues to mean selected packets rejected by a full buffer.
Unselected observations are counted in ``processed`` but not ``selected``.
The temporary scene fields and invalid observations are cleared every step,
so an old score or outage cannot leak into a later opportunity.

The fault definitions are suitable for ``FaultDomain`` and ``FaultSample``;
the tests verify seeded staged/unstaged runs and simultaneous payload faults.
Risk/FMEA aggregation, fitted failure probabilities and an external classifier
remain separate increments. No orbital, power, attitude or radio model is added.

Tests
-----

After the repository's editable installation, run both example modules::

    python -m pytest --noconftest -o addopts= --import-mode=importlib examples/planetary_observation/test_model_main.py examples/planetary_observation/test_model_disturbances.py

The tests compare draws with a separate NumPy generator, verify exact boundary
and injected-fault outcomes, and check packet conservation, density arithmetic,
seed replay, copy/pickle continuation, reset and invalid parameters. They do not
infer statistical accuracy from a few sampled missions.

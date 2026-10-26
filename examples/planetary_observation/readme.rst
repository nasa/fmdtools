Planetary Observation Satellite
===============================

This example provides a deterministic mission scaffold for
`the risk-quantification demonstration in issue #279
<https://github.com/nasa/fmdtools/issues/279>`_. It is a starting point for
planetary/Earth-observation studies using fmdtools, with four interacting
functions and explicit mission-count outputs.

Scope and assumptions
---------------------

The six-minute default sequence is a synthetic software reference, not an orbit
or a ground-station prediction. Power and pointing are assumed available. There
is no orbital, attitude, electrical, optical, thermal or radio-frequency model.
Each observation contains a dimensionless scalar signal in [0, 1], not an image.
The onboard selector is an inclusive threshold rule, not a trained classifier;
its selections do not measure detection accuracy.

Scalar bounds use the existing ``threshold_lim``, ``buffer_capacity_lim`` and
``downlink_capacity_lim`` declarations. Sequence and cross-field checks extend
``Parameter.check_lim``. A small ``get_true_fields`` guard rejects fractional
or boolean packet capacities before integer conversion; it does not repeat
the range checks. ``ObservationSatellite.check_values`` checks the example's
timebase before adding functions. Neither framework constructor is overridden.

Storage and transmission are counted in whole packets. The buffer stores a
count, not payload contents or packet identities. All numerical defaults are
illustrative choices for a hand-checkable reference. They are not mission data
or calibrated reliability estimates. No network access, data download, model
weights or additional dependencies are required.

Architecture
------------

``ObservationSatellite`` is a ``FunctionArchitecture``. Its four ``Function``
blocks communicate through native ``Flow`` objects:

* ``schedule`` writes current visibility, scalar signal and ground contact to
  ``scene`` from an immutable ``MissionParam`` schedule.
* ``capture`` reads ``scene`` and writes one current ``observation``. It clears
  validity and content on steps without an observation opportunity.
* ``process`` reads ``observation``, counts processed/selected packets and adds
  selected packets to ``buffer`` when space is available. Overflow increments
  ``dropped`` without exceeding the capacity.
* ``downlink`` reads ``scene`` contact and drains at most ``downlink_capacity``
  packets from ``buffer`` during that step.

The execution order is schedule, capture, process, downlink. Time zero records
initial conditions; dynamic work occurs at positive integer minutes. A selected
packet may be transmitted on its arrival step, but it must fit in the buffer
first. A full buffer can therefore drop a new selection before a contact frees
space in that same step. Tests protect this ordering explicitly.

The scaffold supports a zero start time, one-minute execution/history timestep,
and a nonnegative integer end time. Shorter runs retain the unfinished backlog;
longer runs do not repeat the schedule. Observation and contact times must be
strictly increasing positive integers. Their schedules need not end within a
shortened run. ``signals`` must match ``observation_times`` in length. Buffer and
downlink capacities are nonnegative integers, including zero.

Run the reference case
----------------------

Use the editable development installation described in the repository's
Contributions guide. From the repository root::

    python -m fmdtools_examples.planetary_observation.model_main

Or use the ordinary propagation API::

    from fmdtools.sim import propagate
    from fmdtools_examples.planetary_observation.model_main import ObservationSatellite

    model = ObservationSatellite()
    result, history = propagate.nominal(model)
    print(result)
    print(history["flows.buffer.s.queued"])

The default observations occur at minutes 1, 2, 3 and 4 with signals
0.2, 0.8, 0.1 and 0.9. Threshold 0.5 selects the second and fourth. Ground
contacts at minutes 5 and 6 deliver one packet each. Buffer capacity is four.

.. list-table:: Reference history
   :header-rows: 1

   * - Minute
     - Acquired
     - Selected
     - Queued
     - Downlinked
     - Dropped
   * - 0
     - 0
     - 0
     - 0
     - 0
     - 0
   * - 1
     - 1
     - 0
     - 0
     - 0
     - 0
   * - 2
     - 2
     - 1
     - 1
     - 0
     - 0
   * - 3
     - 3
     - 1
     - 1
     - 0
     - 0
   * - 4
     - 4
     - 2
     - 2
     - 0
     - 0
   * - 5
     - 4
     - 2
     - 1
     - 1
     - 0
   * - 6
     - 4
     - 2
     - 0
     - 2
     - 0

Terminal classification returns opportunities=4, acquired=4, processed=4,
selected=2, downlinked=2, queued=0 and dropped=0. It assigns neither probabilities
nor cost. Every selected packet is accounted for at every step::

    selected == downlinked + queued + dropped

For example, ``ObservationSatellite(p={"contact_times": ()})`` ends with two
queued packets and no downlinked packets. Setting ``buffer_capacity=1`` with the
other defaults drops one packet and downlinks one. These are deterministic
operational cases, not sampled faults.

Tests
-----

The reference and boundary tests live alongside this example, as do the
introductory ``example.ipynb`` and its executable nominal-run cells. To run
the example tests::

    python -m pytest --noconftest -o addopts= --import-mode=importlib examples/planetary_observation/test_model_main.py

The tests cover complete reference histories, packet conservation, zero and
limited capacities, absent/overlapping contacts, threshold boundaries, partial
and extended runs, repeatability, copied/pickled model continuation, reset,
shared-flow identity and rejected invalid configurations.

Extension boundaries
--------------------

The variants below add seeded disturbances, faults, risk/FMEA analysis and an
offline image classifier without changing this deterministic baseline.
``ProcessObservation.select_observation`` remains the local selection boundary.
A reusable external-model interface remains separate work under
`issue #277 <https://github.com/nasa/fmdtools/issues/277>`_. Real planetary imagery,
calibrated sensor behaviour and mission risk/cost definitions remain outside
these synthetic examples; neither parent issue is closed by this scaffold.

Files
-----

* ``model_main.py`` contains parameters, flows, function blocks, the architecture
  and a nominal-run entry point.
* ``readme.rst`` describes the assumptions, reference outputs and extension
  boundaries.
* ``test_model_main.py`` checks reference trajectories, invariants and invalid
  model inputs without moving example-specific tests into the library suite.
* ``example.ipynb`` steps through the nominal model and two capacity scenarios.

Disturbance and fault variant
----------------------------

The deterministic scaffold stays unchanged. The optional
:doc:`disturbance and fault variant <disturbances>` adds seeded sensor noise,
temporary observation/contact losses and explicit persistent fault modes.
Its tests also live in this example folder.

.. toctree::
   :hidden:

   disturbances
   risk_analysis
   demo_risk_analysis
   classification

Risk and FMEA workflow
---------------------

The :doc:`risk/FMEA workflow <risk_analysis>` samples the declared faults,
compares paired seeded runs and separates conditional packet-loss severity
from fault-weighted contributions. Run ``script_risk_analysis.py`` for a
hand-checkable reference, or open ``demo_risk_analysis.ipynb`` for the seeded
experiment, native FMEA plots and export checks. ``test_risk_analysis.py``
contains the corresponding reference and workflow tests.

Image-classification variant
----------------------------

The optional :doc:`image-classification workflow <classification>` embeds a
scikit-learn estimator using bundled digit images as an offline proxy. It
records detection errors separately from unavailable observations and delivery
counts, with an example-local FMEA script and tests.

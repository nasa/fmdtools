External model adapter pattern
==============================

This example defines a small, synchronous boundary between a native fmdtools
``Function`` and code that evaluates a model. It supports the approach requested
in `issue #277 <https://github.com/nasa/fmdtools/issues/277>`_. The emphasis is on
ownership of inputs, state and errors, with a shippable analytical reference.

The existing :doc:`satellite classifier <../planetary_observation/classification>`
is the concrete third-party integration. The reference here uses only Python
arithmetic so its behavior can be checked without training a model, downloading
data or executing a separate program. It is not a physical simulator or a new
core-library API. The request/response types are local examples to adapt to a
particular model; they are not a universal schema for all external tools.

Run the reference
-----------------

From an editable fmdtools development installation::

    python -m fmdtools_examples.external_simulation.model_main

Or use the normal propagation API::

    from fmdtools.sim import propagate
    from fmdtools_examples.external_simulation.model_main import CoupledFunction

    result, history = propagate.nominal(CoupledFunction())
    assert result["tend.classify.value"] == 6.0
    assert list(history["s.value"]) == [0.0, 2.0, 4.0, 6.0]

The reference integrates a constant rate: ``next_value = value + rate * dt``.
The defaults are rate 2 units/second, an initial value of zero, and updates at
seconds 1, 2 and 3. Time zero records the initial value without advancing it.
An injected ``stopped`` fault supplies a zero rate from its injection time::

    model = CoupledFunction()
    result, history = propagate.one_fault(model, model.name, "stopped", time=2.0)
    faulty = result.get_faulty()
    assert faulty["tend.classify.value"] == 2.0

The fault probability 0.01 per simulation is synthetic. It is used by native
fault sampling; an external call does not independently draw or apply it.

The call contract
-----------------

The wrapper performs four operations in order:

1. Read parameters, flows, state and already-sampled disturbances. Apply the
   relevant fault effects and construct a plain-data snapshot.
2. Validate the request and call the external calculation once.
3. Validate the entire response before changing output state or flows.
4. Commit the validated fields, allowing native histories and ``classify`` to
   record them for analysis.

``StepRequest`` holds the following finite real scalars. Boolean values,
strings, arrays and nonfinite values are rejected by this example's boundary.

.. list-table:: Reference request
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Meaning
   * - ``time``
     - End time of this update in seconds, at least ``dt``.
   * - ``dt``
     - Positive execution-step duration in seconds, taken from ``self.t.dt``.
   * - ``value``
     - Previous accumulated value, copied from native State.
   * - ``rate``
     - Change per second after applying this step's faults and disturbances.

``StepResult.value`` is the proposed next value. ``call_step`` checks its type
and finiteness, including overflow in an otherwise valid calculation. It
returns a fresh scalar result. Frozen dataclasses and scalar normalization
prevent accidental mutation through these request/response objects. This
boundary does not sandbox Python or protect against a deliberately malicious
backend. A backend still must honor the documented no-hidden-state contract.

For an array-based model, define the shape, dtype, ordering, units and valid
range explicitly. Pass an owned array or an immutable representation, rather
than a writable view into a live Flow/State. Validate all output arrays and
copy any backend-owned buffers before committing them. The scalar reference
does not implement or claim that array-copy policy for other adapters.

State and execution ownership
-----------------------------

FMDTools owns the simulation clock, ``Parameter``, ``State``, ``Mode``, ``Rand``,
fault sampling and result/history storage. The backend receives none of those
objects. It must not retain mutable simulation state between calls. All state
needed for the next evaluation is part of the request and response. Calling
the same request twice must give the same result, regardless of call order.

Use ``dynamic_behavior`` when the call advances a model over a time interval.
A native dynamic step is not repeated when the same model time is requested
again. Static behaviors can be reevaluated during propagation, so a stateful
external advance must not be hidden inside them. This example supports a
zero-origin clock with seconds as its units. Execution can be finer than the
history recording interval: ``self.t.dt``, not ``self.sp.dt``, determines the
integration duration. The tests cover that distinction.

``RateRand`` demonstrates native stochastic ownership. With
``sp={"run_stochastic": True}``, a uniform multiplier in [0.9, 1.1) changes the
request rate. Native auto-update draws at time zero too, while the reference
value is still unchanged. Faults do not change the disturbance draw schedule,
so nominal and faulty scenarios remain paired for a fixed seed. Latent random
input densities are not additional Monte Carlo weights.

Copies, staged runs, reset and pickle continuation work here because mutable
state stays in native containers and the backend is stateless. A running
process, socket, mutable cached solver or closure with hidden state does not
satisfy this contract. Such a backend requires a separate, explicit snapshot,
restore, reset and cleanup design before claiming these operations work.
A successful pickle of a wrapper alone does not prove that an external process
has been restored correctly.

Replacing the calculation
-------------------------

The model-specific substitution point is a callable with the signature
``StepRequest -> StepResult``. Define it at module level and bind it as a
``staticmethod`` on a Function subclass. This preserves the selected backend
through the native ``new`` and copy paths without a custom constructor::

    from fmdtools_examples.external_simulation.model_main import (
        CoupledFunction, StepResult,
    )

    def half_rate_step(request):
        return StepResult(request.value + 0.5 * request.rate * request.dt)

    class HalfRateFunction(CoupledFunction):
        external_step = staticmethod(half_rate_step)

    result, _ = propagate.nominal(HalfRateFunction())
    assert result["tend.classify.value"] == 3.0

The test module includes an alternate backend and exercises copy, reset and
pickle continuation with it. Defining a class only inside an interactive
session does not make it importable by a new worker process; parallel workers
should import backend functions and model classes from modules.

Prepare fixed resources through the existing ``init_block`` hook or a
backend-specific factory outside propagation. A cached trained estimator may
be shared for read-only inference when its behavior is independent of prior
calls. Avoid training, downloads or changing shared model weights inside a
simulation step. Do not put a mutable fitted estimator in a Parameter merely
to make it available to the Function.

Failure semantics
-----------------

Malformed inputs fail before backend execution. Backend exceptions propagate
with their original cause; malformed outputs fail before the wrapper commits
its output fields. There are no automatic retries, stale-value substitutions
or guessed output values. Execution errors therefore cannot silently appear
as successful nominal simulation results.

The commit guarantee applies to ``CoupledFunction.s`` in this example.
FMDTools may already have advanced time, drawn disturbances or injected a fault
before a backend raises. Other blocks may also have run. This is not a rollback
of the whole simulation. Discard a failed run or restore a separately verified
checkpoint before retrying; do not continue from an arbitrary failed step.

A modeled failure is different from a tool/transport error. The injected
``stopped`` actuator is a valid modeled input and returns a valid result. When
an external model has meaningful failure statuses, define their allowed values
and explicitly map each to existing Mode/State fields after validating the
whole response. Do not treat every exception as a physical fault, dynamically
accept arbitrary fault names, or include execution errors as successful FMEA
samples. The scalar reference intentionally has no status-to-fault mapping.

Mapping the satellite integration
---------------------------------

The :doc:`satellite classification variant <../planetary_observation/classification>`
uses the same ownership separation with a different, image-specific schema:

* ``ImageCapture`` applies availability, noise, bias and clipping before inference.
* ``predict_digit`` validates a normalized 64-vector or 8-by-8 image, converts it
  to the estimator's input scale, and obtains a class label from scikit-learn.
* ``ClassifyObservation`` decides whether that predicted class is selected and
  retains native buffer and processing-outage behavior.
* The architecture evaluates against held-out truth after inference. Those
  labels do not enter the classifier's decision inputs.

This is a mapping of responsibilities, not a claim that the existing satellite
code uses ``StepRequest`` or ``call_step``. Its estimator-specific assumptions
remain its own contract. A later adapter implementation can add a replaceable
predictor and validate its output shape/class vocabulary without imposing this
accumulator's numeric schema. Bundled digits remain a software proxy rather
than evidence of planetary detection performance.

Adapting an executable later
----------------------------

Keep the same sequence of snapshot, call, validate and commit. A separate
executable adapter would serialize the complete request into an explicit
versioned format, run the selected program, and decode/validate its response.
Specify field units, shape and identity, output limits and supported status
values. Check a request identifier when a transport can mix responses.

Use an explicit argument vector without a shell, a bounded timeout, checked
exit status and a separate working directory for each call or scenario.
Define cleanup for timeouts and child processes; preserve diagnostic context
without accepting partial output as success. Avoid shared filenames across
parallel or staged runs. A timeout on a Python function cannot be enforced by
this synchronous reference; that requires a separate execution mechanism.
No subprocess runner, binary, service, retry policy or process-snapshot support
is shipped here. These are requirements for a later executable example under
issue #277, not capabilities of ``call_step``.

Verification
------------

Run the example-local contract and lifecycle tests::

    python -m pytest --noconftest -o addopts= --import-mode=importlib examples/external_simulation/test_model_main.py

The tests cover exact analytical histories, explicit input validation, invalid
output rejection, backend errors, substitute implementations, negative/zero
rates, seed zero, fractional execution steps, native fault sampling, paired
stochastic inputs, repeated-time calls, copy/reset/pickle and staged/unstaged
agreement. No external process is launched.

For a real backend, additionally compare adapter outputs against direct calls
to that backend and independently known reference cases. Verify process/thread
safety, numerical tolerances and deployment-specific resource cleanup rather
than inferring them from these tests. This example defines a reviewable pattern;
it does not complete the broader external-simulation workstream.

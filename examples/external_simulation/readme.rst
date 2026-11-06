Interfacing fmdtools models with external simulations
=====================================================

Imagine a reservoir filled by a pump. The pump adds two litres per second;
if it stops, the reservoir volume stays constant. This example shows how an
fmdtools ``Function`` calls a separate calculation to update that volume.
The calculation is a small Python function that you can replace with your own
simulator or numerical library.

The files have separate jobs:

* ``external_sim.py`` contains the calculation and its input/output types.
* ``adapter.py`` checks the data exchanged with that calculation.
* ``model_main.py`` connects the adapter to an fmdtools ``Function``.
* ``test_model_main.py`` verifies the result and the simulation lifecycle.

Step 1. Create the external calculation
----------------------------------------

The function in ``external_sim.py`` takes the time, step duration
(``dt``), previous volume in litres and pump rate in litres per second. It returns the
updated value. You can call it without creating an fmdtools model:

.. code-block:: python

    from fmdtools_examples.external_simulation.external_sim import (
        StepRequest,
        reference_step,
    )

    request = StepRequest(time=1.0, dt=1.0, value=0.0, rate=2.0)
    assert reference_step(request).value == 2.0

When adapting this example, replace ``reference_step`` with a function that
calls your own simulator. Give it the inputs it needs and return the result
in a ``StepResult``.

Step 2. Check inputs and outputs
---------------------------------

The ``call_step`` function in ``adapter.py`` validates the request, calls
the chosen calculation once and checks its response:

.. code-block:: python

    from fmdtools_examples.external_simulation.adapter import call_step

    assert call_step(request).value == 2.0

Here the input numbers must be finite, ``dt`` must be positive and the output
must contain a finite value. An invalid response or a failure in the external
calculation raises an exception. Change these checks for the input and output
formats of your own simulator.

Step 3. Connect the calculation to fmdtools
--------------------------------------------

``model_main.py`` defines ``CoupledFunction``. Its native fmdtools
``State`` stores the current value, ``Parameter`` sets the nominal rate,
``Mode`` can stop the actuator and ``Rand`` can vary the rate. Its
``dynamic_behavior`` passes a snapshot of those values to the adapter:

.. code-block:: python

    def dynamic_behavior(self):
        rate = 0.0 if self.m.has_fault("stopped") else self.p.rate * self.r.s.multiplier
        response = call_step(
            StepRequest(self.t.time, self.t.dt, self.s.value, rate),
            self.external_step,
        )
        self.s.put(value=response.value, last_rate=rate, calls=self.s.calls + 1)

This is the part to adapt in your ``Function``: collect its inputs, call the
adapter and write the returned values to native state or flows. The simulation
clock and fault history remain with fmdtools. The calculation itself does not
need to know about fmdtools containers.

Step 4. Run the model and verify the result
--------------------------------------------

With an editable fmdtools installation, run the example from the repository root::

    python -m fmdtools_examples.external_simulation.model_main

You can also run the nominal and faulty cases directly:

.. code-block:: python

    from fmdtools.sim import propagate
    from fmdtools_examples.external_simulation.model_main import CoupledFunction

    model = CoupledFunction()
    result, history = propagate.nominal(model)
    assert list(history["s.value"]) == [0.0, 2.0, 4.0, 6.0]

    result, history = propagate.one_fault(model, model.name, "stopped", time=2.0)
    assert result.get_faulty()["tend.classify.value"] == 2.0

Run the included tests::

    python -m pytest --noconftest -o addopts= --import-mode=importlib examples/external_simulation/test_model_main.py

The tests check the numerical result, invalid inputs and outputs, backend
errors, fault and random inputs, and repeated or staged simulation runs.

Using another external calculation
----------------------------------

Provide a callable that accepts ``StepRequest`` and returns ``StepResult``.
For example, a different calculation can be selected without editing the
fmdtools class:

.. code-block:: python

    from fmdtools_examples.external_simulation.external_sim import StepResult

    def half_rate_step(request):
        return StepResult(request.value + 0.5 * request.rate * request.dt)

    class HalfRateFunction(CoupledFunction):
        external_step = staticmethod(half_rate_step)

    result, _ = propagate.nominal(HalfRateFunction())
    assert result["tend.classify.value"] == 3.0

Put the callable in an importable module for parallel runs. If your backend
retains hidden state or uses a running process, add explicit reset and cleanup
handling before using model copies or staged simulations.

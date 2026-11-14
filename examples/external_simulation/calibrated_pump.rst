Calibrate a reservoir pump with scikit-learn
============================================

The pump in the :doc:`reservoir example <readme>` delivers its commanded
flow. Here, synthetic measurements show that a command of 4 litres per second
delivers 3.2. Fit a regression to those measurements and use it to predict how
much water enters or leaves the reservoir.

From the repository root, install the examples with Python 3.12 or later::

    python -m pip install -e ".[examples]"

Fit the pump curve
------------------

Each measurement pairs one commanded flow with one delivered flow, both in
litres per second. The fitted line passes through zero so a stopped pump
delivers no water. A negative command reverses the flow.

>>> from fmdtools_examples.external_simulation.pump_calibration import fit_pump
>>> estimator = fit_pump()
>>> [round(float(rate), 1) for rate in estimator.predict([[-2.0], [0.0], [2.0]])]
[-1.6, 0.0, 1.6]

Use the prediction in a reservoir update
----------------------------------------

The adapter sends one command to the estimator and multiplies the predicted
flow by the step duration. Starting with 5 litres, a reverse command for two
seconds leaves 1.8 litres. To try your own fitted estimator, pass it as the
``estimator`` argument below. Its ``predict`` method must return one finite
flow for one row containing the commanded flow.

>>> from functools import partial
>>> from fmdtools_examples.external_simulation.adapter import call_step
>>> from fmdtools_examples.external_simulation.external_sim import StepRequest
>>> from fmdtools_examples.external_simulation.pump_adapter import pump_step
>>> reverse = StepRequest(time=2.0, dt=2.0, value=5.0, rate=-2.0)
>>> round(call_step(reverse, partial(pump_step, estimator=estimator)).value, 1)
1.8

Run the pump and inject a stop
------------------------------

``CalibratedPump`` fits the curve when it builds the model and reuses it at
each step. The default command is 2 litres per second, so the volume increases
by 1.6 litres every second.

>>> from fmdtools.sim import propagate
>>> from fmdtools_examples.external_simulation.calibrated_pump import CalibratedPump
>>> model = CalibratedPump()
>>> _, nominal = propagate.nominal(model)
>>> [round(float(v), 1) for v in nominal["s.value"]]
[0.0, 1.6, 3.2, 4.8]

Compare the final volumes when the pump stops at one or two seconds.
The stop takes effect before the update at the injection time. At two seconds
only the first delivery has reached the reservoir.

>>> from fmdtools_examples.external_simulation.pump_scenarios import sample_pump_stops
>>> sample, results, _ = sample_pump_stops(model)
>>> sorted((float(s.time), round(float(results[s.name + ".tend.classify.value"]), 1))
...        for s in sample.scenarios())
[(1.0, 0.0), (2.0, 1.6)]

With a synthetic stop probability of 0.01 per run and 25% of stops at one
second, the expected shortfall is 0.036 litres per run
(0.0025 × 4.8 + 0.0075 × 3.2).

To run the simulation with your own calibration data, write a zero-argument
function that returns a fitted estimator and supply it on a subclass::

    class MyPump(CalibratedPump):
        fit_estimator = staticmethod(my_fit_function)

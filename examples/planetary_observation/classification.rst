Image classification in the satellite workflow
==============================================

This variant places a scikit-learn classifier in the onboard processing
function. It uses the bundled handwritten-digit images as an offline proxy:
these are not planetary images and the results do not establish Earth-observation
accuracy. The original scalar models and their risk example remain unchanged.

Install the repository with its existing examples extra::

    python -m pip install -e '.[examples]'

The model uses ``sklearn.datasets.load_digits`` and ``sklearn.svm.SVC``.
It needs no network access after package installation, no downloaded model
weights, and no separately maintained estimator implementation.

Data and inference contract
---------------------------

The dataset has 1,797 eight-by-eight digit images. As in scikit-learn's
`digit-classification example
<https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html>`_,
an ``SVC(gamma=0.001)`` is fitted to the first 898 rows. Only the remaining
899 rows can be selected by mission ``image_indices``. The partition and model
configuration are fixed independently of disturbance seeds. This is a software
reference split, without claims about unseen spacecraft imagery or writers.

``heldout_digits()`` returns normalized pixels and evaluation labels.
``predict_digit(pixels)`` accepts one finite 64-vector or eight-by-eight image
in [0, 1], restores the estimator's original [0, 16] pixel scale, and returns
a predicted integer class. Labels never enter this call. Invalid input raises;
there is no fallback class or invented confidence score.

The private estimator is fitted once per process during block setup and reused
for read-only inference. Mission copies and fault scenarios do not retrain it.
The scikit-learn dependency is loaded only when this optional variant is used.
See the `dataset documentation
<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>`_
for its origin and format.

Mission behaviour
-----------------

``ClassifyingSatellite`` retains schedule, capture, process and downlink order.
``ImageCapture`` produces pixels for the scheduled held-out row. Existing noise
and payload bias become a single additive brightness offset shared by all
pixels, clipped to [0, 1]. This is common-mode brightness corruption, not
independent pixel noise, optical blur, or a calibrated sensor model.

Obscuration and payload failure suppress acquisition. Processing failure skips
inference. In both cases the prediction is -1, identifying an unavailable
classification rather than a negative detection. Invalid and idle observations
clear their pixels/prediction, preventing stale decisions from being replayed.
Downlink failure retains the selected backlog as before.

``ClassifyObservation`` calls ``predict_digit`` and selects a packet when its
predicted class belongs to ``target_digits``. The inherited scalar ``signals``
and ``threshold`` parameters do not decide selection in this variant. Scalar
``observation.signal`` is recorded as mean image brightness for inspection.

The architecture evaluates predictions only after all mission functions run.
Evaluation labels affect counters, never the classifier, buffer or downlink.
The buffer still stores counts only, so this example cannot determine which
true targets were actually downlinked after drops or limited contacts.

Reference run
-------------

.. code-block:: python

    from fmdtools.sim import propagate
    from fmdtools_examples.planetary_observation.model_classifier import ClassifyingSatellite

    model = ClassifyingSatellite()
    result, history = propagate.nominal(model)
    assert result['tend.classify.true_positives'] == 2
    assert result['tend.classify.true_negatives'] == 2
    assert result['tend.classify.classification_errors'] == 0
    assert result['tend.classify.downlinked'] == 2
    print(history['fxns.process.s.predicted_label'])

The four default held-out labels and predictions are 8, 8, 4, 9. With target
class 8, the first two packets are selected and eventually downlinked. This
four-image reference checks integration; it is not an accuracy benchmark.
Set ``r={'seed': 0}`` and ``sp={'run_stochastic': True}`` for seeded disturbances.
All distribution settings and fault rates retain their illustrative status.

Interpreting counts
-------------------

``true_positives``, ``false_positives``, ``true_negatives`` and
``false_negatives`` describe binary target selection among classified arrivals.
``classification_errors`` instead counts incorrect ten-class digit labels.
``unclassified`` records missed acquisition/processing; its target subset is
``unclassified_targets``. A skipped target is not counted as an inference error.

``missed_targets = false_negatives + unclassified_targets`` counts scheduled
targets that were not selected. It excludes selected packets subsequently lost
to buffer or downlink limitations. The following identities hold at each step:

* processed = true positives + false positives + true negatives + false negatives;
* opportunities = processed + unclassified;
* target opportunities = true positives + missed targets;
* selected = true positives + false positives = downlinked + queued + dropped.

Risk and FMEA
-------------

.. code-block:: python

    from fmdtools_examples.planetary_observation.script_classification import run_classification_analysis

    result, history, sample, fmea = run_classification_analysis(params={'bias': 0.0})
    table = fmea.as_table(sort_by='expected_missed_targets')
    print(table[['sum_scenario_rate', 'average_missed_targets', 'expected_missed_targets']])
    assert abs(table['expected_missed_targets'].sum() - 0.01) < 1e-12

This reuses Task 10's fault catalogue and injection-time validation. Native
FMEA average columns condition on injection time; expected columns apply
scenario rates once. With zero bias and randomness off, each capture/processing
failure contributes 0.01 * (0.25 * 2 + 0.75 * 0) = 0.005 missed targets.
Downlink failure contributes zero to this detection metric despite delivering
zero packets. This is why detection and delivery must be assessed separately.

``stochastic=True`` produces one paired noisy realization for the given seed.
It does not estimate a Monte Carlo mean by itself. Nominal background outcomes
remain available separately. The non-exhaustive fault catalogue is not a total
mission-failure probability model, and input probability densities are not
used as extra weights.

Run and test
------------

From an installed checkout::

    python -m fmdtools_examples.planetary_observation.script_classification
    python -m pytest --noconftest -o addopts= --import-mode=importlib examples/planetary_observation/test_model_classifier.py

Tests cover the fixed training/mission separation, direct scikit-learn
predictions, deliberate errors, truth isolation, input rejection, brightness
corruption, inference suppression, paired stochastic paths, staging, array
ownership, copy/pickle/reset and independent FMEA reference values.

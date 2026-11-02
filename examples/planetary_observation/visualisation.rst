Visual walkthrough
==================

Start with :doc:`demo_satellite` for the complete example: mission delivery,
seeded fault contributions and offline image classification. The notebook
reuses the existing model and analysis modules; it does not define another
simulator or change the numerical calculations.

The five figures and their tables are saved in the notebook so they can also
be inspected without executing it. All values are generated from the retained
simulation outputs. The example uses synthetic schedules and fault rates;
handwritten digits are a software-integration proxy, not planetary imagery.

Run from the repository checkout
--------------------------------

Use a checkout containing this example and select the same Python environment
for installation and the notebook kernel. From the repository root::

    python -m pip install -e ".[examples]" jupyterlab nbconvert ipykernel
    python -m jupyter lab examples/planetary_observation/demo_satellite.ipynb

Choose Restart Kernel and Run All. No dataset or model download is required
once dependencies are installed. The default notebook run writes no output
files. It records package versions and exposes the disturbance seeds and image
bias near the start.

For headless execution, create a new output directory, then run::

    python -c "from pathlib import Path; Path('satellite-demo-output').mkdir()"
    python -m jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=180 --output demo_satellite.executed --output-dir satellite-demo-output examples/planetary_observation/demo_satellite.ipynb
    python -m jupyter nbconvert --to html --output-dir satellite-demo-output satellite-demo-output/demo_satellite.executed.ipynb

This keeps the committed notebook unchanged. The notebook's own optional
export cell can separately save all five figures and seven exact-value CSV
files. Set ``export_directory`` to another new directory. Existing directories
are rejected to avoid overwriting previous results.

Reading the figures
-------------------

.. list-table:: Figure meanings and source outputs
   :header-rows: 1
   :widths: 25 40 35

   * - Figure
     - Meaning
     - Source
   * - Delivery timeline
     - Cumulative delivered packets under three deterministic configurations.
       Backlog and dropped packets remain separate in the table.
     - ``ObservationSatellite`` histories via ``History.plot_line``.
   * - Delivery FMEA
     - Fault-rate-weighted missing packets for the selected single-fault
       catalogue. Conditional injection-time weights sum to one per fault.
     - ``run_risk_analysis`` and native ``FMEA.as_plot``.
   * - Seed variation
     - One complete catalogue contribution per disturbance seed. Its spread
       is not a confidence interval or uncertainty in the supplied fault rates.
     - Per-seed ``Result`` metrics via ``Result.plot_metric_dist``.
   * - Observed images
     - Actual logged pixels for the no-fault and brightness-bias realizations.
       Truth and predictions are reported separately in a small table.
     - ``ClassifyingSatellite`` observation and prediction histories.
   * - Detection FMEA
     - Weighted missed targets at onboard selection, including unavailable
       target observations. Delivered target identities are not represented.
     - ``run_classification_analysis`` and native ``FMEA.as_plot``.

The deterministic reference downlinks two packets; a one-packet buffer delivers
one and drops one; an absent contact schedule leaves two queued. The delivery
FMEA reference sums to 0.045 weighted missing packets. Assertions check these
references, packet conservation, seed-mean reconciliation, classification
accounting and the plotted values.

Keep the interpretation boundaries
----------------------------------

``average_shortfall`` describes severity conditional on an injected fault.
``expected_shortfall`` includes the synthetic fault rate. The no-fault background
shortfall is reported separately. These quantities cannot simply be added to
claim a total mission probability: joint faults and an exhaustive occurrence
model are absent.

For classification, false negatives count rejected targets that received a
prediction. ``unclassified_targets`` counts target opportunities blocked by
acquisition or processing loss. ``missed_targets`` is their sum. Ten-class
prediction errors differ from binary target detection, and packet delivery
counts do not reveal which target images reached the ground.

The four displayed digit images are a fixed held-out illustration, not a
representative performance estimate. Noise and bias are common-mode brightness
offsets. Thirty-two disturbance seeds demonstrate reproducibility and sampling
variation, without establishing convergence or calibrated mission reliability.

Focused walkthroughs
--------------------

For the nominal reference history, use :doc:`example`. For sampling and
seed-weight details, use :doc:`demo_risk_analysis` and :doc:`risk_analysis`.
The model variants are described in :doc:`disturbances` and
:doc:`classification`.

From the repository root, run the example regression suite with::

    python -m pytest --noconftest -o addopts= --import-mode=importlib examples/planetary_observation

A kernel import error generally means the editable installation and selected
kernel differ, or the checkout predates the example. Reinstall from the intended
checkout rather than inserting machine-specific paths into the notebook.
Notebook execution checks calculations; visual inspection should also confirm
legible labels, visible zero-valued contributions and consistent image scales.

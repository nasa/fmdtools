Water Pump
+++++++++++++++++++++++++++++++++++++++++++++

This is an example of a simple pump model used to demonstrate various capabilities of fmdtools. This includes a tutorial notebook, demonstration of plot capabilities, optimization and stochastic modeling.

Further description and use of the model is given in the provided notebooks:

.. base-gallery::
   :caption: Water Pump
   :tooltip:

   tutorial_fmdtools_basics
   demo_fault_analysis
   tutorial_parallelism
   tutorial_optimization
   tutorial_stochastic_behavior

**References**

Hulse, D, Hoyle, C, Tumer, IY, Goebel, K, & Kulkarni, C. “Temporal Fault Injection Considerations in Resilience Quantification.” Proceedings of the ASME 2020 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 11A: 46th Design Automation Conference (DAC). Virtual, Online. August 17–19, 2020. V11AT11A040. ASME. https://doi.org/10.1115/DETC2020-22154


Package Structure
---------------------------------------------

Models:

- `model_main.py`: Main pump model file

- `model_indiv.py`: Modelling of an individual pump Function outside the context of a FunctionArchitecture.

- `model_stochastic.py`: Modelling of the pump with stochastic behaviors

Scripts and tests:

- `script_parallelism.py`: Some functions for exploring parallel model execution

- `script_jointfaults.py`: Script looking at ways of simulating joint faults in a model

- `test_pump_example`: Examples of tests one might run to verify simulation behavior/setup

- `test_model_stochastic.py`: Tests of fmdtools using the stochastic pump model

- `test_pump.py`: Tests of fmdtools using the pump model

Notebooks:

- `tutorial_fmdtools_basics.ipynb`: Tutorial using pump model.

- `demo_fault_analysis.ipynb`: Demonstration of fault simulation.

- `tutorial_parallelism.ipynb`: Tutorial covering how to use parallel computing.
  
- `tutorial_optimization.ipynb`: Tutorial covering how to use fmdtools optimization methods.

- `tutorial_stochastic_behavior.ipynb`: Tutorial covering how to simulate stochastic models.

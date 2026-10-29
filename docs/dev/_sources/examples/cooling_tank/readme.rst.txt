Cooling Tank
+++++++++++++++++++++++++++++++++++++++++++++

This is a fairly simple model of a cooling tank, inlet valve, and outlet valve. It includes a demonstration of the model and optimization of said model.

Further description and use of the model is given in the provided notebooks:

.. base-gallery::
   :caption: Cooling Tank
   :tooltip:

   demo_tank_model
   paper_jmd_optimization

**References:**

- Irshad, L., Hulse, D., Demirel, H. O., Tumer, I. Y., and Jensen, D. C. (May 3, 2021). "Quantifying the Combined Effects of Human Errors and Component Failures." ASME. J. Mech. Des. October 2021; 143(10): 101703. https://doi.org/10.1115/1.4050402

- Hulse, D., and Hoyle, C. (August 8, 2022). "Understanding Resilience Optimization Architectures: Alignment and Coupling in Multilevel Decomposition Strategies." ASME. J. Mech. Des. November 2022; 144(11): 111704. https://doi.org/10.1115/1.4054993


Package Structure
---------------------------------------------

Models:

- `tank_model.py`: Base tank model.

- `tank_optimization_model.py`: Tank model used for optimization

Scripts and tests:

- `test_tank.py`: Tests various tank behaviors.


Notebooks
/////////////////////////////////////////////

- :doc:`Demo: Cooling Tank Model <demo_tank_model>` uses the :class:`~fmdtools.sim.sample.SampleApproach` class to model human interactions with the modeled system (in `model_main.py`).

- :doc:`Paper: Optimization Architecture Comparison <paper_jmd_optimization>` shows how design and contingency management of a system (in `model_optimization.py`) can be co-optimized with the :class:`~fmdtools.sim.search.ProblemArchitecture` class, as well as external solvers.

The support files include various implementations of the tank model.


- The baseline Tank Model (`model_tank.py`), a dynamical implementation of a human-operated tank system to show how fmdtools can be used to model human errors.

- A demonstration Optimization Tank Model (`model_optimization.py`), along with resilience optimization architectures using :class:`~fmdtools.sim.search.ProblemArchitecture`. 
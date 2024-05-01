Tank Model
---------------------------------------------

The tank example is a fairly simple model of a tank, inlet valve, and outlet valve. This example is shown in the following notebooks.

Models
/////////////////////////////////////////////


- `tank_model.py`: Base tank model.

- `tank_optimization_model.py`: Tank model used for optimization


Scripts and tests:
/////////////////////////////////////////////

- `test_tank.py`: Tests various tank behaviors.


Notebooks
/////////////////////////////////////////////

- `Hold-up Tank Model <../examples/tank/Tank_Analysis.ipynb>`_ uses the :class:`fmdtools.sim.sample.SampleApproach` class to model human interactions with the modeled system (in `tank_model.py`).

- `Tank Optimization <../examples/tank/Tank_Optimization.ipynb>`_ shows how design and contingency management of a system (in `tank_optimization_model.py`) can be co-optimized with the :class:`fmdtools.sim.search.ProblemArchitecture` class, as well as external solvers.

The support files include various implementations of the tank model.


- The baseline Tank Model (`tank_model.py`), a dynamical implementation of a human-operated tank system to show how fmdtools can be used to model human errors.

- A demonstration Optimization Tank Model (`tank_opt.py`), resilience optimization architectures using :class:`fmdtools.sim.search.ProblemArchitecture`. 
 
- The main Tank optimization model (`tank_optimization.py`) similar to `tank_model` and `tank_opt` and is a dynamical implementation of a tank system with contingency management. 

.. toctree::
   :hidden:
   
   ../examples/tank/Tank_Analysis.ipynb
   ../examples/tank/Tank_Optimization.ipynb
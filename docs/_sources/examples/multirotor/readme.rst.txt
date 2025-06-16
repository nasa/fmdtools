Overview
---------------------------------------------

Multiple drone models to demonstrate various capabilities of fmdtools.

Models
/////////////////////////////////////////////


- `drone_mdl_static.py`: Base drone model file with very simple static behaviors

- `drone_mdl_dynamic.py`: Extended drone model with dynamic behavior

- `drone_mdl_hierarchical.py`: Extended drone model with component architectures

- `drone_mdl_rural.py`: Extended drone model flying in a rural environment

- `drone_mdl_urban.py`: Extended drone model flying in an urban environment

Scripts and tests:
/////////////////////////////////////////////

- `test_multirotor.py`: Tests various drone behaviors


Notebooks
/////////////////////////////////////////////

The multirotor example model has several models of drones modeled at differing levels of detail which are then used in the following example notebooks;

- :doc:`fmdtools Paper Demonstration <Demonstration>`  is helpful for understanding how a model can be matured as more details are added, covering:

  - The :mod:`~fmdtools.analyze.graph` module.
  
  - Basic simulation of dynamic and static models using methods in :mod:`~fmdtools.sim.propagate` and usage of class :class:`~fmdtools.sim.sample.SampleApproach` for fault sampling
  
  - Analysis using Basic analysis/results processing capabilities
 
- The `Urban Drone Model <drone_mdl_urban.py>`_ is helpful for understanding how to set up gridworlds using :class:`~fmdtools.define.object.coords.Coords` and an Environment class. :doc:`Urban Drone Demo <Urban_Drone_Demo>` demonstrates how this gridworld can be used in simulation.
 
- :doc:`Multirotor Optimization <Multirotor_Optimization>` shows how the design, operations, and contingency management of a system can be co-optimized with the :class:`~fmdtools.sim.search.ProblemArchitecture` class. 

- The support files include various implementations of the drone model.

  - `drone_mdl_static.py` is the baseline model of the drone for static modeling that is used in the other models.

  - `drone_mdl_dynamic.py` expands on the static model to allow for dynamic simulation. It generates behavior-over-time graphs and dynamic/phase-based FMEAs. 

  - `drone_mdl_hierarchical.py` is used to compare system architectures. First by seeing how faults effect the behaviors in each architecture, then by seing how it affects the overall system resilience.

  - `drone_mdl_opt.py` is a modified version of the hierarchical done that encompasses autonomous path planning, rotors, electrical system, and control of the drone. It is parameterized with the following parameters: The rotor and battery architecture can be changed, the flight height can be changed to support different heights, which in turn changes the drone's flight plan, and there is now a `ManageHealth` function which reconfigures the flight depending on detected faults.

References
/////////////////////////////////////////////

- Hulse, D., Walsh, H., Dong, A., Hoyle, C., Tumer, I., Kulkarni, C., & Goebel, K. (2021). fmdtools: A fault propagation toolkit for resilience assessment in early design. International Journal of Prognostics and Health Management, 12(3). https://doi.org/10.36001/ijphm.2021.v12i3.2954

- Hulse, D., Biswas, A., Hoyle, C., Tumer, I. Y., Kulkarni, C., & Goebel, K. (2021). Exploring architectures for integrated resilience optimization. Journal of Aerospace Information Systems, 18(10), 665-678. https://doi.org/10.2514/1.I010942

- Hulse, D, Zhang, H, & Hoyle, C. "Understanding Resilience Optimization Architectures With an Optimization Problem Repository." Proceedings of the ASME 2021 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 3A: 47th Design Automation Conference (DAC). Virtual, Online. August 17–19, 2021. V03AT03A039. ASME. https://doi.org/10.1115/DETC2021-70985

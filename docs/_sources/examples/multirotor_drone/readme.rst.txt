Multirotor Drone
+++++++++++++++++++++++++++++++++++++++++++++

This example includes several models of drones modelled at differing levels of detail, demonstrating how models can be matured as more details are added and how the system can be co-optimized.

You can view a short overview of the model in :doc:`Overview: Multirotor Drone <demo_overview>`.

Further description and use of the model is given in the provided notebooks:

.. base-gallery::
   :caption: Multirotor Drone
   :tooltip:

   demo_overview
   tutorial_fmdtools_basics
   paper_ijphm_fmdtools
   demo_urban_flight
   demo_optimization_architectures


**References:**

- Hulse, D., Walsh, H., Dong, A., Hoyle, C., Tumer, I., Kulkarni, C., & Goebel, K. (2021). fmdtools: A fault propagation toolkit for resilience assessment in early design. International Journal of Prognostics and Health Management, 12(3). https://doi.org/10.36001/ijphm.2021.v12i3.2954

- Hulse, D., Biswas, A., Hoyle, C., Tumer, I. Y., Kulkarni, C., & Goebel, K. (2021). Exploring architectures for integrated resilience optimization. Journal of Aerospace Information Systems, 18(10), 665-678. https://doi.org/10.2514/1.I010942

- Hulse, D, Zhang, H, & Hoyle, C. "Understanding Resilience Optimization Architectures With an Optimization Problem Repository." Proceedings of the ASME 2021 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 3A: 47th Design Automation Conference (DAC). Virtual, Online. August 17–19, 2021. V03AT03A039. ASME. https://doi.org/10.1115/DETC2021-70985


Package Structure:
---------------------------------------------
Models:

- `model_static.py`: Base drone model file with very simple static behaviors

- `model_dynamic.py`: Extended drone model with dynamic behavior

- `model_hierarchical.py`: Extended drone model with component architectures

- `model_rural.py`: Extended drone model flying in a rural environment

- `model_urban.py`: Extended drone model flying in an urban environment

Scripts and tests:

- `test_multirotor.py`: Tests various drone behaviors

- `script_opt_model_rural.py`: Functions for optimizing `model_rural.py`.


Notebooks:

-  `demo_overview.ipynb`: an overview of the purpose and uses of this example.

-  `tutorial_fmdtools_basics.ipynb`: basic tutorial using the multirotor model to show how to use fmdtools.

-  `paper_ijphm_fmdtools.ipynb`: demonstration showing how a model can be matured as more details are added.

-  `demo_urban_flight.ipynb`: Usage of the urban drone, which sets up the drone flight over a gridworld representing different buildings.

-  `demo_optimization_architectures.ipynb`: Optimization of the drone.
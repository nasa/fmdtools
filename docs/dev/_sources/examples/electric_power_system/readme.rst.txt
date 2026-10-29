Electric Power System
+++++++++++++++++++++++++++++++++++++++++++++

The EPS model is a model of a simple electric power system in `eps.py`, which shows how undirected propagation can be used in a simple static (i.e., one time-step) modeling use-case. 

Further description and use of the model is given in the provided notebooks:

.. base-gallery::
   :caption: Electric Power System
   :tooltip:

   demo_static_models

**References:**

- Hulse, D, Hoyle, C, Tumer, IY, & Goebel, K. "Decomposing Incentives for Early Resilient Design: Method and Validation." Proceedings of the ASME 2019 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 2B: 45th Design Automation Conference. Anaheim, California, USA. August 18–21, 2019. V02BT03A015. ASME. https://doi.org/10.1115/DETC2019-97466

- Hulse, D, Zhang, H, & Hoyle, C. "Understanding Resilience Optimization Architectures With an Optimization Problem Repository." Proceedings of the ASME 2021 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 3A: 47th Design Automation Conference (DAC). Virtual, Online. August 17–19, 2021. V03AT03A039. ASME. https://doi.org/10.1115/DETC2021-70985

Package Structure:
---------------------------------------------

Models:

- `eps.py`: Only EPS model. The main purpose of this system is to supply power to optical, mechanical, and heat loads. Failure behavior of the system is represented at a high level using solely the functions of the system. 

Scripts and tests:

- `test_model.py`: Tests various EPS behaviors.

Notebooks:

- `demo_static_models.ipynb`: demonstrates this model and some basic fmdtools methods.
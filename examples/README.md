
# Examples and tutorials for using fmdtools

-----------------


This only includes descirptions of the main python files, not the supporting files such as `__init__` for intialization, `test_model.py` functions/files necessary for code control, and resulting outputs \(graphs, excel files, etc\). Notebooks are in bold with inidividual files in italics.

## asg_demo

- **`Action_Sequence_Graph.ipynb`**: Covers the use of the Action Sequence Graph in the FxnBlock class, which is useful for representing a Function's Progress through a sequence of actions \(e.g., modes of operation, etc\).


## eps

- **`EPS_Example_Notebook.ipynb`**: Shows an example replicating previous the simple electric power system implemented in [IBFM](https://github.com/DesignEngrLab/IBFM) in the `eps example` directory, with some basic fault propagation and visualization.

- *`eps.py`*: This electrical power system model showcases how fmdtools can be used for purely static propogation models (where the dynamic states are not a concern). This EPS system was previously provided in the [IBFM fault modelling toolkit](https://github.com/DesignEngrLab/IBFM) and other references--this implementation follows the simple_eps model in IBFM. The main purpose of this system is to supply power to optical, mechanical, and heat loads. In this model, we represent the failure behavior of the system at a high level using solely the functions of the system. Further information about this system (data, more detailed models) is presented at: https://c3.nasa.gov/dashlink/projects/3/


## multiflow_demo

- **`Multiflow_and_Commsflow_Demonstration.ipynb`**: Demonstrates the use of MultiFlow and CommsFlow for the coordination of multiple devices via a single "agent". With the exception of the graphs, the code here is identical to `multiflow_demo.py` with the purpose of walking people through the step-by-step creation of the model.

- *`multiflow_demo.py`*: The model of the demonstration of using MultiFlow and CommsFlow. 


## multirotor

- **`Demonstration.ipynb`**: Uses a high-level model of a multirotor drone to illustrate the following fmdtools model types: Network Model, Static Model, Dynamic Model, and Hierarchical Model. It highlights the uses in the original paper: [Hulse, D., Walsh, H., Dong, A., Hoyle, C., Tumer, I., Kulkarni, C., & Goebel, K. (2021). fmdtools: A Fault Propagation Toolkit for Resilience Assessment in Early Design](https://doi.org/10.36001/ijphm.2021.v12i3.2954). It relies on the models in this same folder, specifically `drone_mdl_static.py`,  `drone_mdl_dynamic.py`, and `drone_mdl_hierarchical.py`.

- **`Multirotor_Optimization.ipynb`**: Uses the drone model defined in `drone_mdl_opt.py` to illustrate the use of the `ProblemInterface` class to set up optimization architectures. Prior to viewing this study, it may be helpful to get some background on the problem and optimization architectures, by reviewing the following references. [Hulse, D., Biswas, A., Hoyle, C., Tumer, I. Y., Kulkarni, C., & Goebel, K. (2021). Exploring Architectures for Integrated Resilience Optimization.](https://doi.org/10.2514/1.I010942) presents a version of this Drone Optimization case study, and also introduces the concept of a resilience optimization architecture. [ Hulse, D., & Hoyle, C. (2022). Understanding Resilience Optimization Architectures: Alignment and Coupling in Multilevel Decomposition Strategies](https://doi.org/10.1115/1.4054993) provides a better review of what is meant by "optimization architectures" as well as different formulations which may be used in this context.

- *`drone_mdl_static.py`*: This is the baseline model of the drone for static modeling that is used in the other models. It uses displaygraph views of fault scenarios and produce a static FMEASee `Demonstration.ipynb` for results.

- *`drone_mdl_dynamic.py`*: This expands on the static model to allow for dynamic simulation. It generates behavior-over-time graphs and dynamic/phase-based FMEAs. See `Demonstration.ipynb` for results.

- *`drone_mdl_hierarchical.py`*: This expands the model to compare system architectures. First by seeing how faults effect the behaviors in each architechture, then by seing how it affects the overall system resilience. See `Demonstration.ipynb` for results.

- *`drone_mdl_opt.py`*: This drone has similar structure and behaviors to the drone in `drone_mdl_hierarchical.py` (see below), encompassing the autonomous path planning, control, rotors, electrical system, and control of the drone. However, this model has been parameterized with the following parameters: The rotor and battery architecture can be changed, the flight height can be changed to support different heights, which in turn changes the drone's flight plan, and there is now a `ManageHealth` function which reconfigures the flight depending on detected faults. See `Multirotor_Optimization.ipynb` for results.



## pump
- **`Pump_Example_Notebook.ipynb`**: Shows basic I/O operations that can be performed with this toolkit, as well as some of the basic model and simulation visualization and analysis features. This script runs these basic operations on the simple model defined in `ex_pump.py`.

- **`Tutorial_unfilled.ipynb`**:  Provide a basic interactive tutorial for learning basic fmdtools functions with the blank cells for students to fill in.

- **`Tutorial_complete.ipynb`**:  The completed version of `Tutorial_unfilled`.

- **`AST_Sampling.ipynb`**: Illustrates running [Adaptive Stress Testing](https://www.nasa.gov/content/tech/rse/research/adastress) on fmdtools model using the AdaStress package. It uses `pump_stochastic.py`.

- **`Optimization.ipynb`**: Shows the basics of setting up a resilience optimization problem with the `Problem` class in `fmdtools.sim.search` module. It uses `pump_stochastic.py`.

- **`Parallelism_Tutorial.ipynb`**: Discusses how to use parallel programming in fmdtools, including, how to set up a model for parallelism, the syntax for using parallelism in simulation functions, and considerations for optimizing computational performance in a model. It uses `ex_pump.py`

- **`Stochastic_Modelling.ipynb`**: Covers the basics of stochastic modelling in fmdtools. This notebook covers how to construct a stochastic model by adding *stochastic states* to function blocks and incorporating them in function behavior, how to simulate single scenarios and distributions of scenarios in `propagate` methods and how to visualize and analyze the results of stochastic model simulations.

- *`ex_pump.py`*: This model constitudes an extremely simple functional model of an electric-powered pump.

- *`pump_stochastic.py`*: A simple model for explaining stochastic behavior modelling. This model is an extension of `ex_pump.py`` that includes stochastic behaviors.


## rover

- **`Approach_Use-Cases.ipynb`**: Covers how to set up a fault sampling approach and use it to simulate a large number of hazardous scenarios in a model.

- **`Model_Structure_Visualization_Tutorial.ipynb`**: . Demonstrates fmdtools' interfaces both for setting up model structures and for fault visualization. It includes a build through of the rover model, demonstration of various graphs, and visualization of running the model.

- **`Nominal_Approach_Use-Cases.ipynb`**: Demonstrates how to evaluate the performance of a model over a set of input parameters. This can be used to define/understand the operational envelope for different system parameters and quantify failure probabilites given stochastic inputs. It covers the build of a rover model.

- **`Rover_Setup_Notebook.ipynb`**: Covers the setup of a lane-following rover fault model for understanding the effects of faults in AI-driven systems. This model uses the fmdtools simulation toolkit to simulate the nominal and faulty behaviors of the rover over a set of fault scenarios and classify/assess risk. It uses `rover_model.py`. 

- **`Degradation_Modelling_Notebook.ipynb`** Shows how degradation modelling can be performed to model the resilience of an engineered system over its entire lifecycle. It uses `rover_model.py` and `rover_model_human.py`.

- **`Rover_Mode_Notebook.ipynb`** Shows how modes can be elaborated from health states in an fmdtools model. It uses `rover_model.py`


- **`IDETC_Human_Paper_Analysis.ipynb`** Support notebook for a pape covering the exploration and analysis of error producing conditions by a human operator in the rover model See [Irshad, L., Hulse, D., Resilience Modeling in Complex Engineered Systems With Human-Machine Interactions](https://doi.org/10.1115/DETC2022-89531) It uses `rover_model.py` and `rover_model_human.py`.

- *`rover_model.py`*: The main rover model defining the functions and flows used in analysis.

- *`rover_degradation.py`*: An extension to the main rover model to include degradated states.

- *`rover_model_human.py`*: An extension to the main rover model to include human faults and responses.

## tank

- **`Tank_Analysis.ipynb`**: Show the basics of using fmdtools to simulate hazards in a system with human-component interactions, including human-induced failure modes, human responses to component failure modes, and joint human-component failure modes.

- **`Tank_Optimization.ipynb`**: Demonstrates the optimization of the tank model described in `tank_optimization_model.py` using the `ProblemInterface` class in the `fmdtools.search` module. This is identical to `tank_opt.py` in functionality.

- *`tank_model.py`*: Dynamical implementation of a human-operated tank system to show how fmdtools can be used to model human errors

- *`tank_opt.py`*: Demonstration of resilience optimization architectures using ProblemInterface. 
 
- *`tank_optimization_model.py`*: Dynamical implementation of a tank system with contingency management. This model is similar to tank_model and tank_opt.



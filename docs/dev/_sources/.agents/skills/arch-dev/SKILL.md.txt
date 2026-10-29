---
name: arch-dev
description: Develop models of systems with multiple behaviors interacting with each other
license: MIT
metadata:
  audience: users
---

## What I do and when to use me

- Create fmdtools models of systems where there are important interactions between system behaviors
- When the `FunctionArchitecture` `ActionArchitecture` or `ComponentArchitecture` classes are to be used.

## Instructions

### Determining type of model

- Be default, use a `FunctionArchitecture` class
- If the model is a sequence of actions `ActionArchitecture` classes may be built out of `Action` classes instead
- Use a `ComponentArchitecture` class if the user specifies

### Function Architectures


- Determine the type of model per [Structuring a Model Best Practices](../../../docs-source/best-practices.md#structuring-a-model). Apply this structure going forward.
- See the relevant code template in [Intro_to_fmdtools.md](../../../docs-source/Intro_to_fmdtools.md#function-architecture-code-template)

- Note that the goal of the functional architecture is to propagate behaviors **between different functions** using flows. If there are no interacting behaviors, there isn't a reason to use a FunctionArchitecture
- Generally, interactions will be executed in the individual function level and propagated between each other. As a result, architecture-level behavior methods are not often used.

#### Simple Systems Models

- Some of the relevant examples would be:
- `examples/electric_power_system`
- `examples/water_pump`

#### Systems Models

- Some of the relevant examples would be:

- `examples/cooling_tank`
- `examples_multirotor_drone`
- `examples/navigating_rover`

#### System of Systems models
- Identify the relevant fmdtools classes:
	- Agents, the System, and Environmental Behaviors would be `Function` classes 
	- The data structure representing the shared environment would inherit from the `fmdtools.define.environment` class with internal `GeometryArchitecture` and `Coords` classes (if needed)
	- Communications would be `CommsFlow` classes while Perceptions would be `MultiFlow` classes and standard shared variables would be `Flow` classes

- Some of the relevant examples of system-of-systems models are in:

	- `examples/airport_taxiway`, for multi-aircraft operations on a shared runway
	- `examples/airspacelib`, for multi-drone operations interacting with an environment, `GeometryArchitecture` and `Coords` classes, and structuring a more complex library of models
	- `examples/navigating_rover`, for a single system operating in a larger environment
	- `examples/state_communication`, for usage of `CommsFlow`


### Action Architectures

- Some of the relevant examples would be:

- `examples/navigating_rover/model_human.py`
- `examples/human_hazard_mitigation/model_main.py`
- `examples/cooling_tank/model_main.py`

### Component Architectures

- Some of the relevant examples would be:

- `examples/multirotor_drone/model_hierarchical.py`

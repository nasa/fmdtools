# dronelib 0.3

The Drone Resilience Modelling Library (dronelib) is a library that can be used to simulate drones interacting with their environment at a high level. The goal of the dronelib package is to enable the rapid simulation of drones in the airspace.

The dronelib package extends the fmdtools modeling and simulation libary with drone-specific classes to represent drone behaviors (e.g., 3-d trajectory, navigation, etc).

It also provides a number of worked examples of drone modelling in a range of use-cases, including drone in-flight contingency management and wildfire response.

## Structure

The dronelib package has the following structure:
- dronelib.base: base classes that can be adapted to new case studies
- dronelib.contingencymanagement: case study modelling a drone that must perform in-flight contingency management to respond to battery depletion and airspace intrusion scenarios
- dronelib.wildfireresponse: case study modelling a set of drones in a wildfire response situation

## Requirements

```
fmdtools v2.2.3
```

## Installation

### Using UV

1.) Install uv and overall development environment

2.) Set up uv virtual environment, e.g. ( `uv venv` and `.venv\Scripts\activate`)

a.) To set up with Spyder, first install Spyder 6 or later.

b.) run `uv pip install spyder-kernels==3.1.*` (or whatever version is required by your version of Python/spyder).

c.) Then the console can be made as the ipython console in Spyder (by setting the console path to `.venv\Scripts\python.exe` in preferences).

3.) Install dronelib

a.) If using a set version of fmdtools, use `uv pip install -e .`

b.) If co-developing with fmdtools, install fmdtools as a development dependency also, e.g., using `uv pip install -e fmdtools @ ../fmdtools`

### Using pip directly

```
   pip install -e "/path/to/dronelib" 
```

## Contributors

Daniel Hulse
Seydou Mbaye
Cody Wang

## Funding

Funded as a part of the NASA Aeronautics Research Mission Directorate's System-Wide-Safety Project.
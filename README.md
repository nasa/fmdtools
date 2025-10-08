# aerialdrm

The Aerial Disaster Response Model (aerialdrm) is a model of aerial disaster response that can be adapted to a variety of disaster use-cases.


## Goals

The goal aerialdrm project is to enable the analysis of the resilience of disaster response missions, including:

- Wildland firefighting

- Hurricane relief and recovery (future)


## Requirements

```
fmdtools v2.1
```

## Installation

### Using UV

1.) Install uv and overall development environment

2.) Set up uv virtual environment, e.g. ( `uv venv aerialdrm-dev` and `aerialdrm-dev\Scripts\activate`)

a.) To set up with Spyder, first install Spyder 6 or later.

b.) run `uv pip install spyder-kernels==3.1.*`.

c.) Then the console can be made as the ipython console in Spyder (by setting the console path to `aerialdrm-dev\Scripts\python.exe` in preferneces).

3.) Install aerialdrm

a.) If using a set version of fmdtools, use `uv pip install -e .`

b.) If co-developing with fmdtools, install fmdtools as a development dependency also, e.g., using `uv pip install -e fmdtools @ ../fmdtools`

### Using pip directly

```
   pip install -e "/path/to/aerialdrm" 
```

## Contributors

Daniel Hulse
Seydou Mbaye
Cody Wang

## Funding

Funded as a part of the NASA Aeronautics Research Mission Directorate's System-Wide-Safety Project.
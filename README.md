# fmdtools: Fault Model Design Tools

fmdtools is a toolkit to enable the modelling and design of systems in the early design phase. With it, one can model the dynamic effects of a fault over time in an engineered system and design the system to be more resilient. 

[![DOI](https://zenodo.org/badge/212862445.svg)](https://zenodo.org/badge/latestdoi/212862445)

##### Features include:

- graph-based fault propagation enabling forward and backward degradation of model flows
- static and dynamic fault propagation
- visualization of the effect of faults on model parameters over time
- visualization of model graph status over time and at given timesteps
- ability to run through a list of scenarios and provide an FMEA-style table of faults, effects, rates, costs, and overall risk
- quantification of average and expected resilience metrics (e.g. expected time degraded for model functions and flows)

In the future, we would like to add features for:

- optimization/design exploration
- uncertainty quantification
- parallelism

WARNING: This is a research code and is currently under development. Some features may not work as desired and may change during development.

----
## Getting Started

### Prerequisites

fmdtools requires Python 3 and depends on these packages:

- `networkx`
- `matplotlib`
- `mpl_toolkits`
- `numpy`
- `scipy`
- `pandas`
- `jupyter notebook`
- `ffmpeg` (for animations)
- `quadpy` (for quadrature sampling)

so make sure to install (e.g. using `pip install packagename` or `conda install packagename`) them before running any of the codes.

One of the example scripts is provided in Jupyter Notebook, so install it if you would like to use that.

### Examples

A simple example model is provided in the `pump example` directory. `ex_pump.py` shows the underlying model classes that must be construted for a given system while `Pump Tutorial` shows how one would use the toolkit for a variety of fault propogation and resilience quantification use-cases.

----
## License

MIT


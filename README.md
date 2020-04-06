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
- `numpy`
- `matplotlib`			(for plots)
- `pandas`				(for tables)
- `jupyter notebook`	(for notebooks)
- `netgraph`	 		(for setting graph node positions)
- `ffmpeg` 				(for animations)
- `quadpy` 				(for quadrature sampling)
- `mpl_toolkits`		(for drone model)

so make sure to install (e.g. using `pip install packagename` or `conda install packagename`) them before running any of the codes.

One of the example scripts is provided in Jupyter Notebook, so install it if you would like to use that.

### Examples

A simple example model is provided in the `pump example` directory. `ex_pump.py` shows the underlying model classes that must be construted for a given system while `Pump Tutorial` shows how one would use the toolkit for a variety of fault propogation and resilience quantification use-cases.

A static model is provided in the `eps example` directory. `eps.py` shows the model (previously implemented in [IBFM](https://github.com/DesignEngrLab/IBFM) and other works) while `EPS Example Notebook.ipynb` shows some basic fault propogation and visualization. 

An example of modelling a dynamical system (without faults) is shown in the `\pandemic example` directory. A reference stand-alone model is provided in `simple_model.py` and `fmd_model.py` and `fmd_model_script.py` show how one might implement a distributed version of this model in `fmdtools`.

`hold-up tank example` provides an example of modelling human-interactions with the modelled system using the `component` class.

`multirotor example` provides an example of representing functions with underlying componenent architectures.

----
## Contributors

[Daniel Hulse](https://github.com/hulsed)

[Hannah Walsh](https://github.com/walshh) : Network analysis codes

[Hongyang Zhang](https://github.com/zhangho2) : Pandemic model

----
## License

MIT


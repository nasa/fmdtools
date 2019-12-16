# fmdtools: Fault Model Design Tools

fmdtools is a toolkit to enable the modelling and design of systems in the early design phase. With it, one can model the dynamic effects of a fault over time in an engineered system and and then design the system to be more resilient based on rate and severity information. 

##### Features include:

- graph-based fault propagation enabling forward and backward degradation of model flows
- propagation of faults in a timeless sense or over time
- visualization of the effect of faults on model parameters over time
- visualization of model graph status over time and at given timesteps
- ability to run through a list of scenarios and provide an FMEA-style table of faults, effects, rates, costs, and overall risk
- quantification of average and expected resilience metrics (e.g. expected time degraded for model functions and flows)

In the future, we would like to add features for:

- probability modelling/uncertainty quantification
- optimization

WARNING: This is a research code and is currently under development. Some features may not work as desired and may change during development.

----
## Getting Started

### Prerequisites

fmdtools requires Python 3 to run. 

It also uses the following packages to be installed on Python 3:

- `networkx`
- `matplotlib`
- `mpl_toolkits`
- `numpy`
- `scipy`
- `pandas`
- `ffmpeg` (for animations)
- `shapely` (for `quad_mdl` only)
- `quadpy` (for quadrature sampling)

so make sure to install (e.g. using `pip install packagename`) them before running any of the codes.

One of the example scripts is provided in Jupyter Notebook, so install it if you would like to use that.

### Examples

A simple example model is provided in `ex_pump.py` and an example of setting up the model, propagating single faults and a list of faults, and displaying results with that model is provided in `pump_script.py`.

A (more complicated) model is provided in `quad_mdl.py` and `quad_script.py` for a small drone.

----
## Contributors
Daniel Hulse

----
## License

MIT


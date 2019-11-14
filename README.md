# FFERMAT: Function-Failure Early Resilience Models And Tools

FFERMAT is a toolkit to enable the modelling and design of systems in the early design phase. With it, one can model the dynamic effects of a fault over time in an engineered system and and then design the system to be more resilient based on rate and severity information. 

##### Features include:

- graph-based fault propagation enabling forward and backward degradation of model flows
- propagation of faults in a timeless sense or over time
- display of a systems behavior in faulty or nominal states
- ability to run through a list of scenarios and provide an FMEA-style table of faults, effects, rates, costs, and overall risk

In the future, we would like to add features for:

- optimization
- quantification of resilience 

WARNING: This is a research code and is currently under development. Some features may not work as desired and may change during development.

----
## Getting Started

### Prerequisites

FFERMAT requires Python 3 to run. 

It also uses the following packages to be installed on Python 3:

- `networkx`
- `matplotlib`
- `mpl_toolkits`
- `numpy`
- `pandas`
- `ffmpeg` (for animations)
- `shapely` (for `quad_mdl` only)

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

----
## Acknowledgements

FFERMAT was originally based on the IBFM code written by Matthew McIntire and available at https://github.com/DesignEngrLab/IBFM.

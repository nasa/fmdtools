Examples
==============================================

This repository provides a several resources to get familiar with fmdtools:

Intro to fmdtools Workshop and Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The intro to fmdtools workshop provides a basic high-level overview of modelling and simulation in fmdtools, with a corresponding model and example notebook to fill out.

To start the workshop, first download the workshop slides (:download:`Intro to fmdtools.pptx <docs/Intro to fmdtools.pptx>`), tutorial model (:download:`ex_pump.py <example_pump/ex_pump.py>`), and Unfilled Notebook :download:`ex_pump.py <example_pump/Tutorial_unfilled.ipynb>`.  If you cloned fmdtools, you can just navigate to these files in the repository--they are in a directory called ``example_pump``. Then, follow along with the slides and the filled-in notebook (see: `fmdtools tutorial <example_pump/Tutorial_complete.ipynb>`_).


Example Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After completing the workshop, it can be helpful to run through the following notebooks to better understand fmdtools modelling, simulation, and analysis basics as well as more advanced features and use-cases:

- `Defining and Visualizing fmdtools Model Structures <docs/Model_Structure_Visualization_Tutorial.ipynb>`_

- `fmdtools Paper Demonstration <example_multirotor/Demonstration.ipynb>`_ (for understanding network/static/dynamic/hierarchical model types)

- `Pump Example Notebook <example_pump/Pump_Example_Notebook.ipynb>`_ (for showcasing plots/tables/visualization capabilities)

- `Defining Fault Sampling Approaches in fmdtools <docs/Approach_Use-Cases.ipynb>`_ (for understanding fault sampling in fmdtools)

- `Using Parallel Computing in fmdtools <example_pump/Parallelism_Tutorial.ipynb>`_ (for simulating fault scenarios in parallel and reducing computational costs)

- `Defining Nominal Approaches in fmdtools <docs/Nominal_Approach_Use-Cases.ipynb>`_ (for simulating the model at different parameters in nominal/faulty scenarios)

- `Stochastic Modelling in fmdtools <example_pump/Stochastic_Modelling.ipynb>`_ (for simulating models with stochastic behavior and/or inputs)

Other Examples

- `Hold-up Tank Model <example_tank/Tank_Analysis.ipynb>`_ Using the `component` class to model human interactions with the modelled system in `hold-up tank example`.

- `EPS Example Notebook <example_eps/EPS_Example_Notebook.ipynb>`_ (shows static modelling)
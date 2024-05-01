Taxiway Model
---------------------------------------------

This provides a model of different aircraft moving on a taxiway. Specifically, this example shows how to model, test, and simulate different agents individually and then compose them in a combined functional architecture.


Models
/////////////////////////////////////////////

- `common.py`, which provides the base methods/classes for the models,

- `asset.py` which models the individual assets (aircraft etc), respectively,

- `ATC.py`, which models the air traffic control,

- `model.py` which models the assets and ATC in a combined architecture.


Scripts and tests:
/////////////////////////////////////////////

- `test_asset.py`, tests asset behaviors

- `test_model.py` tests combined model behaviors

- `propositional_network_testing.py` shows some more advanced setups for working with graphs in the contexts of multiflows.


Notebooks
/////////////////////////////////////////////

A demo is provided in `Paper Notebook <../examples/taxiway/Paper_Notebook.ipynb>`_, which shows basic analysis of this model in two scenarios.


.. toctree::
   :hidden:

   ../examples/taxiway/Paper_Notebook.ipynb
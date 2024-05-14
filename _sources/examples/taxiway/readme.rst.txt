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

A demo is provided in `Paper Notebook <Paper_Notebook.ipynb>`_, which shows basic analysis of this model in two scenarios.

References
/////////////////////////////////////////////

- Irshad, L, & Hulse, D. "Modeling Distributed Situation Awareness in Resilience-Based Design of Complex Engineered Systems." Proceedings of the ASME 2023 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 2: 43rd Computers and Information in Engineering Conference (CIE). Boston, Massachusetts, USA. August 20–23, 2023. V002T02A050. ASME. https://doi.org/10.1115/DETC2023-116689


.. toctree::
   :hidden:

   Paper_Notebook.ipynb
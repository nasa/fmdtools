Overview
---------------------------------------------

This provides a model of different aircraft moving on a taxiway. Specifically, this example shows how to model, test, and simulate different agents individually and then compose them in a combined functional architecture.


Models
/////////////////////////////////////////////

- `model_common.py`, which provides the base methods/classes for the models,

- `model_asset.py` which models the individual assets (aircraft etc), respectively,

- `model_atc.py`, which models the air traffic control,

- `model_main.py` which models the assets and ATC in a combined architecture.


Scripts and tests:
/////////////////////////////////////////////

- `test_asset.py`, tests asset behaviors

- `test_model.py` tests combined model behaviors


Notebooks
/////////////////////////////////////////////

A demo is provided in :doc:`Paper[jcise]: Distributed Situation Awareness <paper_jcise_dsa>`, which shows basic analysis of this model in two scenarios.

References
/////////////////////////////////////////////

- Irshad, L, & Hulse, D. "Modeling Distributed Situation Awareness in Resilience-Based Design of Complex Engineered Systems." Proceedings of the ASME 2023 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 2: 43rd Computers and Information in Engineering Conference (CIE). Boston, Massachusetts, USA. August 20–23, 2023. V002T02A050. ASME. https://doi.org/10.1115/DETC2023-116689


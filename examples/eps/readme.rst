EPS Model
---------------------------------------------

The EPS model is a model of a simple electric power system in `eps.py`, which shows how undirected propagation can be used in a simple static (i.e., one time-step) modeling use-case. 

Models
/////////////////////////////////////////////

- `eps.py`: Only EPS model. The main purpose of this system is to supply power to optical, mechanical, and heat loads. Failure behavior of the system is represented at a high level using solely the functions of the system. 

Scripts and tests:
/////////////////////////////////////////////

- `test_eps.py`: Tests various EPS behaviors.

Notebooks
/////////////////////////////////////////////

- `EPS Example Notebook <../examples/eps/EPS_Example_Notebook.ipynb>`_ demonstrates this model and some basic fmdtools methods. It shows how fmdtools can be used for purely static propagation models where dynamic states are not a concern. This is a replication of a previous simple electric power system implemented in `IBFM <https://github.com/DesignEngrLab/IBFM>_`.


.. toctree::
   :hidden:
   
   ../examples/eps/EPS_Example_Notebook.ipynb
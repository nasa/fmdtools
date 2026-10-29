# Exploring Faulty-State Space using MOO

## Getting Started

The three pertinent scripts are **ccea.py**, **ea.py**, and **random.py**. There are accessory scripts that externally handle some analysis of results, **animation.py** and **ea_analysis.py**, but these are secondary.

**ccea.py** is the Cooperative Co-evolutionary Algorithm. **ea.py** is the
standard Evolutionary Algorithm. **random.py** is the Random Sampling Algorithm.

## Evolutionary Algorithms

**ccea.py** and **ea.py** were both written with the help of **DEAP**, a python
framework for evolutionary algorithms. Below are links to resources for the
framework.

[DEAP Documentation](https://deap.readthedocs.io/en/master/index.html)

[DEAP GitHub](https://github.com/deap/deap)


### CCEA

For the CCEA, there were no examples in the **DEAP** documentation that perfectly modeled this project, but ["A Funky One"](https://deap.readthedocs.io/en/master/tutorials/basic/part1.html#a-funky-one) proved useful for guidance. Additionally, [DEAP GitHub COEV Examples](https://github.com/DEAP/deap/tree/master/examples/coev) were helpful as well.

Moveover, the preexisting mutation, crossover and selection functions in **DEAP** were not directly implemented but, instead, used as a guide to build the functions for this project.

`varAnd2()` is the evolutionary algorithm that employs `mutHealthStates()` and `cxHealthStates()` for mutation and crossover, respectively.

Here, a population of size `SUBPOP_SIZE * NUM_SUBPOP` consists of  subpopulations, such that a subpopulation is defined as a list of faulty-space tuples (called individuals in the code for this isolated case) of size `SUBPOP_SIZE`. A solution is a representative which is a list of individuals such that one individual is selected from each subpopulation, resulting in a list of `NUM_SUBPOP` number of faulty-state space tuples.

When running an experiment, the parameters to vary are:
- `SUBPOP_SIZE`
: number of individuals per subpopulation
- `NUM_SUBPOP`
: number of subpopulations
- `ngen`
: number of generations

Then, the top `k` number of individuals are selected. Currently, the CCEA is using a tournament selection. `k` and the selection function can be changed here:

```
toolbox.register("select", tools.selTournament,k=int(0.5*SUBPOP_SIZE), tournsize=SUBPOP_SIZE)
```

### EA

The EA was developed after the CCEA and is a simplified version of the CCEA, so if the CCEA makes sense, then the EA _should_ be easier to comprehend. It,  also, closely resembles many examples in the [DEAP GitHub Examples](https://github.com/DEAP/deap/tree/master/examples) if one wishes for extra resources.

Similarly to the CCEA, `varAnd2()` is the evolutionary algorithm that employs `mutHealthStates()` and `cxHealthStates()` for mutation and crossover, respectively.

Here, a population of size `POP_SIZE` consists of individuals, such that an individual is defined as a list of faulty-state space tuples of size `IND_SIZE`. A solution in an individual and it is found after `ngen` number of generations.

When running an experiment, the parameters to vary are:
- `IND_SIZE`
: number of faulty-state space tuples per individual
- `POP_SIZE`
: number of individuals in population
- `ngen`
: number of generations

Then, the top `k` number of individuals are selected. Currently, the EA is using a `selBest` selection. `k` and the selection function can be changed here:

```
toolbox.register("select", tools.selBest,k=int(0.5*POP_SIZE))
```

### Random

The Random Sampling Algorithm is similar in structure to the EA code. A population is randomly generated and an individual from that population is randomly selected per generation. Its fitness values are computed and, if the preceding generation performs worse, then it replaces the preceding generation's individual. Otherwise, it is thrown out.
The variables are defined in the same way as the EA.

When running an experiment, the parameters to vary are:
- `IND_SIZE`
: number of faulty-state space tuples per individual
- `POP_SIZE`
: number of individuals in population
- `ngen`
: number of generations


## Results and Analysis

The results from **ccea.py**, **ea.py**, and **random.py** are either stored in a **csv** as Pandas Dataframe (and then interpreted) or generated as a **pdf** or **png** after compiling each program.

The dataframe is then analyzed in **animation.py** and **ea_analysis.py**.

**ea_analysis.py** plots the fitness values of the top performing individual(or representative in the case of CCEA) in each generation for all algorithm.

_Note: if one runs into any errors, consider checking the_ **csv** _result files_ (**rslt_ccea.csv**,**rslt_ea.csv**,**rslt_random.csv**). _Errors typically arise in the there after running each of the algorithms._ 

[![fmdtools logo - titanic](/docs/figures/logo/logo-titanic.png)](https://github.com/nasa/fmdtools)

# Overview

[![PyPI](https://img.shields.io/pypi/v/fmdtools)](https://pypi.org/project/fmdtools/)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/nasa/fmdtools?label=GitHub%20Release)](https://github.com/nasa/fmdtools/releases)
[![GitHub Tag](https://img.shields.io/github/v/tag/nasa/fmdtools)](https://github.com/nasa/fmdtools/tags)
[![CodeFactor](https://www.codefactor.io/repository/github/nasa/fmdtools/badge)](https://www.codefactor.io/repository/github/nasa/fmdtools)
[![Tests Status](./tests-badge.svg)](https://htmlpreview.github.io/?https://github.com/nasa/fmdtools/blob/main/reports/junit/report.html)
[![Tests Coverage](./coverage-badge.svg)](https://htmlpreview.github.io/?https://github.com/nasa/fmdtools/blob/main/reports/coverage/index.html)
[![GitHub License](https://img.shields.io/badge/License-NOSA-green)](https://github.com/nasa/fmdtools/blob/main/NASA_Open_Source_Agreement_fmdtools.pdf)
[![NASA Software Classification](https://img.shields.io/badge/Software_Class-E-blue)](https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_7150_002D_&page_name=AppendixD)

**fmdtools** (Fault Model Design tools) is a Python library for modelling, simulating, and analyzing the resilience of complex systems. With fmdtools, you can (1) represent system structure and behavior in a model, (2) simulate the dynamic effects of hazardous scenarios on the system, and (3) analyze the results of simulations to understand and improve system resilience.

[Click here to view the the full documentation website.](https://nasa.github.io/fmdtools)

## About

The fmdtools libary provides the computational support needed to evolve towards a simulation-based (rather than document-based) hazard analysis process that [**enables the consideration of systems resilience**](https://nasa.github.io/fmdtools/docs/Development%20Guide.html#why-fmdtools). This means that it can be used to extend the scope of hazard analysis from component faults to the dynamic interactions between the system, operators, and the environment. Some key features include:

<img align="left" width="100" height="100" src="/docs/figures/powerpoint/flexible.svg">

### Flexible Modelling Paradigm
Models in fmdtools use a consistent and composable representation of system structure and behavior. Whether you want to model a simple component, a complex system-of-systems, or both, fmdtools can help.

<img align="left" width="100" height="100" src="/docs/figures/powerpoint/powerful.svg">

### Powerful Simulation Techniques
Simulation techniques in fmdtools represent the state-of-the-art in dynamical systems modelling for resilience quantification. With fmdtools, you can simulate the dynamic effects of hazardous scenarios over a wide range of variables to quantify and optimize risk, resilience, and safety metrics. 

<img align="left" width="100" height="100" src="/docs/figures/powerpoint/efficient.svg"> 

### Efficient Analysis Process
Readily-deployable analysis methods are built in to fmdtools to enable the rapid and iterative statistical analysis of simulation results. With fmdtools, you can write 2-3 lines of code to visualize model behavior instead of spending hours writing it yourself.



## Getting Started

### Set up python environment

The fmdtools library was developed to run in an anaconda python environment. If you do not have an existing python environment, first [download and install anaconda.](https://docs.anaconda.com/free/anaconda/install/index.html).

After the base installation, install these external external (non-python) packages using anaconda:

```
jupyter notebook
# used for: repository notebooks
# license: (BSD-3) https://jupyter.org/governance/projectlicense.html
# install from: (should be installed already)
graphviz
# used for: plotting graphs using graphviz
# license: (CPL 1.0) https://graphviz.org/license/ 
# install from: https://anaconda.org/anaconda/graphviz
ffmpeg
# used for: animations in demo notebook(s)
# license: (LGPL version) https://www.ffmpeg.org/legal.html
# install from: https://anaconda.org/conda-forge/ffmpeg
```

### Install fmdtools

The latest public version of fmdtools can be downloaded from the [fmdtools github repository](https://github.com/nasa/fmdtools/) e.g., using:

```
   git clone https://github.com/nasa/fmdtools.git
```
   
For development and use of this version (e.g., for tutorials and models), we recommended installing this package using `pip`:

```
   pip install -e "/path/to/fmdtools" 
```

A version of the fmdtools toolkit can also be installed directly from the [PyPI package repository](https://pypi.org/project/fmdtools/) using ``pip install fmdtools``.


#### Dependencies

fmdtools requires Python 3 (anaconda recommended) and depends directly on these packages (see requirements.txt):

```
scipy
# license: (BSD-new) https://www.scipy.org/scipylib/license.html
tqdm
# license: (mixed) https://github.com/tqdm/tqdm/blob/master/LICENCE
networkx
# license: (BSD-new) https://raw.githubusercontent.com/networkx/networkx/master/LICENSE.txt
numpy
# license: (BSD) https://numpy.org/doc/stable/license.html
matplotlib
# license: (mixed) https://matplotlib.org/stable/users/license.html
pandas
# license: (BSD 3-clause) https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html#license
ordered-set
# license: (MIT) https://github.com/rspeer/ordered-set/blob/master/MIT-LICENSE
dill 
# license: (MIT) https://github.com/uqfoundation/dill/blob/master/LICENSE
recordclass == 0.21.1
# license: (MIT) https://github.com/intellimath/recordclass/blob/main/LICENSE.txt
pytest
# license: (MIT) https://docs.pytest.org/en/7.3.x/license.html
graphviz
# license: (MIT) https://github.com/xflr6/graphviz/blob/master/LICENSE.txt
shapely
# license: (BSD 3-clause) https://github.com/shapely/shapely/blob/main/LICENSE.txt
```

Since these are direct dependencies, they will be installed automatically along with fmdtools.

Some additional indirect dependencies may be needed for development/testing of the code, or for specific notebooks. Thus, to develop/contribute to fmdtools, it can be helpful to install these up-front using `pip install packagename`:
```
deap
# used for optimization of rover faults
# license: (LGPL-3.0) https://github.com/DEAP/deap/blob/master/LICENSE.txt
pymoo
# used for optimization of tank example
# license: (Apache 2.0) https://github.com/anyoptimization/pymoo/blob/main/LICENSE
sklearn
# used for rover mode notebook
# license: (BSD-3 Clause) https://github.com/scikit-learn/scikit-learn?tab=BSD-3-Clause-1-ov-file#readme
nbmake
# used for notebook tests
# license: (Apache 2.0) https://github.com/treebeardtech/nbmake/blob/main/LICENSE
pytest-html  
# used for development test report generation
# license: (MPL-3) https://github.com/pytest-dev/pytest-html/blob/master/LICENSE
coverage
# used for measuring test coverage
# license: (Apache 2.0) https://github.com/nedbat/coveragepy/blob/master/NOTICE.txt
pytest-cov
# used for measuring test coverage
# license: (MIT) https://github.com/pytest-dev/pytest-cov/blob/master/LICENSE
genbadge
# used for generating test badges for the README
# license: (BSD 3-Clause) https://github.com/smarie/python-genbadge/blob/main/LICENSE
multiprocess
# used for parallism tutorial profiling
# license: (BSD-3 Clause) https://github.com/uqfoundation/multiprocess/blob/master/LICENSE
pathos          
# used for parallelism tutorial profiling
# license: (BSD-3 Clause) https://github.com/uqfoundation/pathos/blob/master/LICENSE
```

One `fmdtools` is installed, you should be able to run:
```
import fmdtools
```

To check the version of fmdtools, you can run the following:
```
import importlib.metadata
importlib.metadata.version("fmdtools")
```
which should return the current version of fmdtools.

If a development install has been performed, you can further check aspects of your installation by running `run_all_tests.py` and opening the corresponding test report in `/reports/junit/report.html` to see if all tests pass (or, are consistent with the [current test report](https://github.com/nasa/fmdtools/blob/main/reports/junit/report.html)).

### Explore Tutorials and Resources

Once fmdtools is installed, use the following to get acquainted with how to use libary:

- Go through the [Intro to fmdtools workshop](https://nasa.github.io/fmdtools/docs/Intro_to_fmdtools.html) to learn about some of the basics of the fmdtools library and work with an existing model. 

- Explore more [examples](https://nasa.github.io/fmdtools/examples/Examples.html) of particular use-cases by going through the [examples folder](https://github.com/nasa/fmdtools/tree/main/examples)

- Read about contributions and model development best practices by perusing the [Development Guide(https://nasa.github.io/fmdtools/docs/Development%20Guide.html#).

- Explore the searchable [module reference](https://nasa.github.io/fmdtools/docs/fmdtools.html) for syntax and usage documentation.

## Contributions
fmdtools is developed primarily by researchers at NASA Ames Research Center. External contributions are welcome under a Contributor License Agreement:

- [Individual CLA](https://github.com/nasa/fmdtools/blob/main/fmdtools_Individual_CLA.pdf)

- [Corporate CLA](https://github.com/nasa/fmdtools/blob/main/fmdtools_Corporate_CLA.pdf)

### Contributors

<a href="https://github.com/nasa/fmdtools/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nasa/fmdtools" />
</a>

See: [CONTRIBUTORS.md](CONTRIBUTORS.md)

## Citing this repository

To cite fmdtools in general, you may cite our explanatory publication:

```
@article{hulse2021fmdtools,
  title={fmdtools: A fault propagation toolkit for resilience assessment in early design},
  author={Hulse, Daniel and Walsh, Hannah and Dong, Andy and Hoyle, Christopher and Tumer, Irem and Kulkarni, Chetan and Goebel, Kai},
  journal={International Journal of Prognostics and Health Management},
  volume={12},
  number={3},
  year={2021}
}
```

To cite a particular version of the fmdtools, you may use:

```
@software{nasa2024fmdtools,
  author = {{NASA}},
  title = {fmdtools},
  url = {https://github.com/nasa/fmdtools},
  version = {2.0-rc3}, # <- replace with your version number
  date = {2024-05-01},
}
```

To cite a particular fmdtools example or published research methodology, use the relevant reference provided in the in accompanying README file for the example.

## License/Notices

Released under the [NASA Open Source Agreement Version 1.3](https://github.com/nasa/fmdtools/blob/main/NASA_Open_Source_Agreement_fmdtools.pdf)

Copyright © 2024 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.


### Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

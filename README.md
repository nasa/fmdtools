# Overview

![PyPI](https://img.shields.io/pypi/v/fmdtools)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/nasa/fmdtools?label=GitHub%20Release)
[![CodeFactor](https://www.codefactor.io/repository/github/nasa/fmdtools/badge)](https://www.codefactor.io/repository/github/nasa/fmdtools)
[![Tests Status](./tests-badge.svg)](./reports/junit/report.html)
[![GitHub License](https://img.shields.io/badge/License-NOSA-green)](https://github.com/nasa/fmdtools/blob/main/NASA_Open_Source_Agreement_fmdtools.pdf)

**fmdtools** (Fault Model Design tools) is a toolkit for modelling system resilience in the early design phase. With it, one can simulate the effects of faults in a system to build resilience into the system design at a high level.  To achieve this, fmdtools provides a Python-based *design environment* where one can represent the system in a model, simulate the resilience of the model to faults, and analyze the resulting model responses to iteratively improve the resilience of the design.

Note: This version (**2.0-alpha4**) is currently in development, and thus not all interfaces may be fully stable and not all examples or documentation may be up-to-date. For stable versions, download packages releases without the -alpha or -beta tags.

[Click here to view the the full documentation website.](https://nasa.github.io/fmdtools)

## Getting Started

The latest public version of fmdtools can be downloaded from the [fmdtools github repository](https://github.com/nasa/fmdtools/) e.g., using:

```
   git clone https://github.com/nasa/fmdtools.git
```
   
For development and use of this version (e.g., for tutorials and models), we recommended then installing this package using `pip`:

```
   pip install -e "/path/to/fmdtools" 
```

A version of the fmdtools toolkit can also be installed directly from the [PyPI package repository](https://pypi.org/project/fmdtools/) using ``pip install fmdtools``.


### Prerequisites

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
recordclass >=0.14.4
# license: (MIT) https://bitbucket.org/intellimath/recordclass/src/master/LICENSE.txt
pytest
# license: (MIT) https://docs.pytest.org/en/7.3.x/license.html
graphviz
# license: (MIT) https://github.com/xflr6/graphviz/blob/master/LICENSE.txt
```

These external (non-python) packages are recommended to enable specific fmdtools use-cases and to test/develop fmdtools as well as work with examples in the repository:

```
jupyter notebook
# used for: repository notebooks
# license: (BSD-3) https://jupyter.org/governance/projectlicense.html
graphviz
# used for: plotting graphs using graphviz
# license: (CPL 1.0) https://graphviz.org/license/ 
pyvis
# used for: interactive html views of model graphs
# license: (BSD-3)
ffmpeg
# used for: animations in demo notebook(s)
# license: (LGPL version) https://www.ffmpeg.org/legal.html
```

These must be installed within the system environment (e.g. using ``conda install packagename``) them to enable the specific features/uses in the repository. 

Additionally, the following python packages are not included but are necessary for development/testing of the code:
```
shapely
# used for: multirotor model
# license: (BSD 3-clause) https://github.com/shapely/shapely/blob/main/LICENSE.txt
deap
# used for: optimization of rover faults
# license: (LGPL-3.0) https://github.com/DEAP/deap/blob/master/LICENSE.txt
nbmake
# used for: for notebook tests
# license: (Apache 2.0) https://github.com/treebeardtech/nbmake/blob/main/LICENSE
pytest-html  
# used for: development test report generation
# license: (MPL-3) https://github.com/pytest-dev/pytest-html/blob/master/LICENSE
genbadge
# used for: generating test badges for the README
# license: (BSD 3-Clause) https://github.com/smarie/python-genbadge/blob/main/LICENSE
multiprocess
# used for: parallism tutorial profiling
# license: (BSD-3 Clause) https://github.com/uqfoundation/multiprocess/blob/master/LICENSE      
pathos          
# used for: parallelism tutorial profiling
# license: (BSD-3 Clause) https://github.com/uqfoundation/pathos/blob/master/LICENSE 
```
They can be installed using ``pip install packagename``.

## Examples

For tutorials, see the [examples folder](https://github.com/nasa/fmdtools/tree/main/examples) and [examples page of the documentation](https://nasa.github.io/fmdtools/docs/Examples.html). These folders include the following: 

- [**asg_demo**](https://github.com/nasa/fmdtools/tree/main/examples/asg_demo): A tutorial covering the use of the Action Sequence Graph in the FxnBlock class, which is useful for representing a Function's Progress through a sequence of actions \(e.g., modes of operation, etc\).

- [**eps**](https://github.com/nasa/fmdtools/tree/main/examples/eps): A model of a simple electric power system in eps.py, which shows how undirected propagation can be used in a simple static (i.e., one time-step) moelling use-case.

- [**multiflow_demo**](https://github.com/nasa/fmdtools/tree/main/examples/multiflow_demo): A demonstration on the use of MultiFlow and CommsFlow for the coordination of multiple devices.

- [**multirotor**](https://github.com/nasa/fmdtools/tree/main/examples/multirotor): Includes several models of drones modelled at differing levels of detail. Includes a demonstration of how models can be matured as more details are added and how the system can be co-optimized.

- [**pump**](https://github.com/nasa/fmdtools/tree/main/examples/pump): A simple pump model to demonstrate various capabilities of fmdtools. This includes a tutorial notebook, demostration of plot capabilities, optimization and stochastic modeling.

- [**rover**](https://github.com/nasa/fmdtools/tree/main/examples/rover): Showcases more advanced methodologies that can be used in fmdtools, and has essentially been the developers’ demo case study for advancing the state-of-the-art in resilience simulation. 

- [**tank**](https://github.com/nasa/fmdtools/tree/main/examples/tank): A fairly simple model of a tank, inlet valve, and outlet valve. It includes a demonstration of the model and optimization of said model.

## Contributions
fmdtools is developed primarily by researchers at NASA Ames Research Center. External contributions are welcome under a Contributor License Agreement:

- [Individual CLA](https://github.com/nasa/fmdtools/blob/main/fmdtools_Individual_CLA.pdf)

- [Corporate CLA](https://github.com/nasa/fmdtools/blob/main/fmdtools_Corporate_CLA.pdf)

### Contributors

<a href="https://github.com/nasa/fmdtools/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nasa/fmdtools" />
</a>

See: [CONTRIBUTORS.md](https://github.com/nasa/fmdtools/blob/main/CONTRIBUTORS.md)

## License/Notices

Released under the [NASA Open Source Agreement Version 1.3](https://github.com/nasa/fmdtools/blob/main/NASA_Open_Source_Agreement_fmdtools.pdf)

Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.


### Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

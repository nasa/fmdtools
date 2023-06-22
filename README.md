# Overview

![PyPI](https://img.shields.io/pypi/v/fmdtools)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/nasa/fmdtools?label=GitHub%20Release)


**fmdtools** (Fault Model Design tools) is a toolkit for modelling system resilience in the early design phase. With it, one can simulate the effects of faults in a system to build resilience into the system design at a high level.  To achieve this, fmdtools provides a Python-based *design environment* where one can represent the system in a model, simulate the resilience of the model to faults, and analyze the resulting model responses to iteratively improve the resilience of the design.

Note: This version (**2.0-alpha2**) is currently in development, and thus not all interfaces may be fully stable and not all examples or documentation may be up-to-date. [Click here to view the status of our test results.](https://htmlpreview.github.io/?https://github.com/nasa/fmdtools/blob/main/pytest_report.html). For stable versions, download packages releases without the -alpha or -beta tags.

## Getting Started

The latest public version of fmdtools can be downloaded from the [fmdtools github repository](https://github.com/nasa/fmdtools/) e.g., using:

```
   git clone https://github.com/nasa/fmdtools.git
```
   
For development and use of this version (e.g., for tutorials and models), we recommended then installing this package using `pip`:

```
   pip install -e /path/to/fmdtools 
```

A version of the fmdtools toolkit can also be installed directly from the [PyPI package repository](https://pypi.org/project/fmdtools/) using ``pip install fmdtools``.


### Prerequisites

fmdtools requires Python 3 and depends directly on these packages (see requirements.txt):

```
   scipy
   tqdm
   networkx
   numpy
   matplotlib
   pandas
   ordered-set
   dill 
   recordclass >=0.14.4
   pytest
```

These packages are optional but recommended to enable specific fmdtools use-cases and to work with examples in the repository:

```
   jupyter notebook			#(for repository notebooks)
   graphviz					#(to plot using graphviz options)
   pyvis					#(for interactive html views of model graphs)
   ffmpeg 					#(for animations)
   shapely					#(for multirotor model)
   deap						#(for optimization of rover faults)
   pycallgraph2				#(for model profiling)
```

These must be installed (e.g. using ``pip install packagename`` or ``conda install packagename``) them before running any of the codes in the repository. 


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

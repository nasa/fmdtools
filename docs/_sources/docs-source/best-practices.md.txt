## Pay attention to and document the fmdtools version

As a research-oriented tool, much of the fmdtools interfaces can be considered to be "in development." While we want to keep the repository stable, there have been many changes to syntax over the years to provide enhanced functionality and usages.

As such, it can be helpful to document what fmdtools version you are running in a README.md for your project, so you can always have a working model and replicate results, even if something has changed on the toolkit-level. 

This also helps us (as developers) address bugs which affect specific versions of fmdtools, as well as identify bugs which were introduced in updates.


## Plan your model to avoid technical debt

Simple, small models are relatively easy to define in fmdtools with a few functions, flows, and behaviors. As such, it can be easy to get in the habit of not planning or organizing development systematically, which leads to issues when developing larger models. Specifically, code that is *written into existence* instead of designed, planned, edited, tested, and documented. This leads to [Technical debt](https://en.wikipedia.org/wiki/Technical_debt), which is the inherent difficulty of modifying code that was written ad-hoc rather than designed. Unless this technical debt is resolved, the ability to modify a model (e.g., to add new behaviors, conduct analyses, etc) will be limited by the complicated and unwieldy existing code. 

The next subsections give some advice to help avoid technical debt, based on lessons learned developing fmdtools models over the past few years.

## Make an FRDL Diagram First

The FRDL ontology was designed to help define system functional architectures in a way that comports with simulation in fmdtools.

As such, as a part of planning what a model is supposed to represent, it can be helpful to define the system in FRDL first, to fully outline the scope, behaviors, and key interactions of the model will be. This diagram can help make key system architecture decisions as well, keep track of what the overall goal-state of the model is to be, and help keep track of certain interaction details that can be helpful when explaining and debugging simulations.

## Use model constructs to simplify your code

The fmdtools codebase is quite large, and, as a result, it can be tempting to dive into modeling before learning about all of its capabilities. The problem with this is that many of these capabilities and interfaces are there to make your life easier, provided you understand and use them correctly. Below are some commonly-misunderstood constructs to integrate into your code:

* `fmdtools.define.container.base.BaseContainer` has a number of very basic operations which can be used in all containers to reduce the length of lines dedicated solely to assignment and passing variables between constructs. Using these methods can furthermore enable one to more simply perform vector operations with reduced syntax.
* `fmdtools.define.object.timer.Timer` can be used very simply to represent timed behavior and state-transitions. 
* While modes can be used to describe fault modes in a very general way, faulty behavior that can also be queried from the model using the concept of a *disturbance*, which is merely a change in a given variable value. While disturbances are less general, they require much less to be implemented in the model. Disturbances can be passed as an argument (as a dict or as a part of a Sequence class) to :`fmdtools.sim.propagate.sequence()`
* `fmdtools.define.container.parameter.Parameter` and parameter-generating functions are helpful for understanding the model operating envelope. In general, try to avoid having parameters that duplicate each other in some way.
* Randomness can be used throughout, but use the specified interfaces (`fmdtools.define.container.rand.Rand`, etc.) so that a single seed is used to generate all of the rngs in the model. Not using these interfaces can lead to not being able to replicate results consistently.
* A variety of custom attributes can be added to `fmdtools.define.block.function.Function` and `fmdtools.define.flow.base.Flow`, but not every custom attribute is going to work with staged execution and parallelism options. In general, use containers to represent things that change and parameters to represent things that don't change. If you want to do something fancy with data structures, you may need to re-implement `fmdtools.define.block.base` methods for copying and returning states to `propagate`.
* If there's something that you'd like to do in an fmdtools model that is difficult with existing model structures, consider filing a bug report before implementing an ad-hoc solution. Alternatively, try developing your solution as a *feature* rather than a hack to solve a single use-case. If the features is in our scope and well-developed, we may try to incorporate it in our next release.


## Don't copy, inherit and functionalize

Copy-and-paste can be a useful concept, but often gets over-relied upon by novice model developers who want to create several variants of the same programming structure. However, in the world of systems engineering (and software development), there are many cases where developers should be using [class inheritance](https://www.w3schools.com/python/python_inheritance.asp) and [writing functions](https://ucsbcarpentry.github.io/2019-10-10-Python-UCSB/14-writing-functions/) instead. 

The advantages of inheritance are: 

1. It reduces the bulk amount of code required to represent functions, making code more comprehensible.
2. It makes the distinction between similar classes easier to distinguish, since *there is no redundant code*.
3. It makes it easier to edit code afterward, since the developer *only has to edit in one place* for it to apply to all the relevant types.
4. It makes testing easier, since common methods only need to be tested once.

In fmdtools, these patterns can be helpful:

* Instead of creating two very similar `fmdtools.define.block.function.Function` classes (e.g. Drone and PilotedAircraft) and copying code between each, create a single class (e.g. Aircraft) with common methods/structures (e.g., Fly, Taxi, Park, etc.) and then use sub-classes to extend and/or replace methods/structures in the common class as needed (e.g., Autonomous Navigation in the Drone vs. Piloted Navigation in the normal aircraft).
* Instead copying code for the same operations to several different places in a model, write a single function instead. This method can then be documented/tested and extended to a variety of different use-cases which require the same basic operation to be done. 

This is an incomplete list. In general, it can be a helpful limitation to *try to avoid using copy-and-paste as much as possible.* Instead if a piece of code needs to be run more than once in more than once place, write a function or method which will be used everywhere. The idea should be to *write the code once.*

## Document your code, sometimes *before* your write it

In general, Python coding style aspires to be [as self-documenting](https://en.wikipedia.org/wiki/Self-documenting_code) as possible. However, this is not a replacement for documentation. In general, novice developers think of documentation as something which happens at the end of the software development process, as something to primarily assist users. 

This neglects the major benefits of documentation in the development process. Specifically:

1. it helps other people understand how to use and *contribute to* your code,
2. it helps define what your code is supposed to do, its interfaces, and the desired behavior, and
3. as a result, it helps you understand your code.

For fmdtools models, documentation should at the very least take the following form:

* A README (in markdown or rst) that explains how to set up the model environment (e.g., requirements/dependencies), as well as explains the structure of the folder/repo (model file, tests, analyses, etc.)
* Documented examples of using the code. Usually you can use a jupyter notebook for this to show the different analyses you can run with your model.
* Docstrings document the classes and functions which make up your model. These are most important for development and should include:
	* An overall module description (top of file)
	* Docstrings for flows: What does the state represent? What are the states? What values may these take?
	* Docstrings for `fmdtools.define.block.function.Function`: What are the states, parameters, behaviors, and modes?
	* For any method/function, try to follow existing docstring conventions, with a summary of the purpose/behavior of the method, and a description of all input/output data types.

Documentation can best be thought of as a *contract that your code should fulfill*. As such, it can be very helpful to think of the documentation first, as a way of specifying your work. Tests (formal and informal) can then be defined based on the stated behavior of the function. It is thus recommended to *document your code as you write it*, instead of waiting until the end of the development process, to avoid technical debt. 

## Don't get ahead of yourself--try to get a running simulation first


In the model development process, it can often be tempting to try to model every single mode or behavior in immense detail from the get-go. This is motivated by a desire to achieve realism, but can lead to issues from a project management and integration perspective. A model does not have much meaning outside a simulation or analysis, and, as such, development needs to be motivated *first* by getting a working simulation and *then* by adding detail. These simulations are the key feedback loop for determining whether model code is embodying desired behavior. 

A very basic model development process should thus proceed:

1. Create Architecture file and create place-holder major function/flow classes
2. Connect classes in a Architecture file and visualize structure
3. Create low-fidelity model behaviors and verify in nominal scenario
4. Add hazard metrics in `classify` 
5. Add more detailed behaviors (e.g., modes, actions, components, etc) as needed
6. Perform more complex analyses...

In general, it is bad to spend a lot of time developing a model without running any sort of simulation for verification purposes. This toolkit has been designed to enable the use of simulations early in the development process, and it is best to use these features earlier rather than later.

Finally, *smaller, incremental iterations are better than large iterations.* Instead of spending time implementing large sections of code at once (with documentation and testing TBD), instead implement small sections of code that you can then document, test, and edit immediately after. Using these small iterative cycles can increase code quality by ensuring that large blocks of undocumented/untested (and ultimately unreliable) code don't make it into your project, only for you to have to deal with it later.

## Preserve your prototype setup by formalizing it as a test

Testing code is something which is often neglected in the development process, as something to do when the project is finished (i.e., as an assurance rather than development task). Simultaneously, developers often iterate over temporary scripts and code snippets during development to ensure that it works as expected in what is essentially an informal testing process. The major problem with this process is that these tests are easily lost and are only run one at a time, making it difficult to verify that code works after it has been modified.

Instead, it is best to *formalize scripts into tests*. This can be done with Python's [unittest](https://docs.python.org/3/library/unittest.html) module, which integrates well with existing python IDEs and enables execution of several different tests in a sequence. Instead of losing prototype code, one can easily place this code into a `test_X` method and use it iteratively in the development process to ensure that the code still works as intended. This is true even for more "qualitative" prototype script, where the output that is being iterated over is a plot of results. Rather than abandoning a prototyping setup like this, (e.g., by commenting it out), a much better approach is to formalize the script as a test which can be run at the will of the user when desired. In this case, the plot should show the analysis and describe expected results so that it can be quickly verified. The testing of plots is enabled with the function `fmdtools.analyze.common.suite_for_plots`, which enables you to filter plotting tests out of a model's test suite (or specify only running specific tests/showing specific plots). 

While testing is an assurance activity, it should also be considered a development activity. Testing ensures that the changes made to code do not cause it to take on undesired behaviors, or be unable to operate with its interfacing functions. To enable tests to continue to be useful through the modelling process, they should be given meaningful names as well as descriptions describing what is being tested by the test (and why).

Finally, do not create tests solely to create tests. Tests should have a specific purpose in mind, ideally single tests should cover as many considerations as possible, rather than creating new tests for each individual consideration. As in model development, try to avoid bloat as much as possible. If the desire is to cover every edge-case, try to parameterize tests over these cases instead of creating individual test methods.

## Edit your code

The nature of writing code is a messy process--often we spend a considerable amount of time getting code to a place where it "works" (i.e., runs) and leave it as-is. The problem with doing this over and over is that it neglects the syntax, documentation, and structural aspects of coding and thus contributes to technical debt. One of the best ways to avoid this from impacting development too much is to edit code after writing it.

Editing is the process of reviewing the code, recognizing potential (functional and stylistic) problems, and ultimately revising the code to resolve these problems. In this process, all of the following concerns should be considered:

* Do the data structures make logical sense? Are they used systematically throughout the project?
* Are operations organized with a logical structure? Is it easy to see what is performed in what sequence? Are lines too long? 
* Are naming and stylistic conventions being followed? Do variables have self-explanatory names? Are names being spelled correctly?
* Are lines too long? Are there too many nested statements?
* Are the methods/classes fully documented? 
* Will the functions work in every possible case implied by the documentation?
* Is inheritance being used correctly? 
* Is the code re-inventing existing fmdtools structure or syntax or going against existing protocols?
* Does it pass all tests?

This is an incomplete list. The point is to regularly review and improve code *after it is implemented to minimize future technical debt*. Waiting to edit will cause more hardship down the line.

## Structuring a model

fmdtools was originally developed around a very simple use-case of modeling physical behaviors using a Function/Flow ontology, where Functions (referred to as "technical functions") are supposed to be the high-level roles to be performed in the system, while flows are the data passed between these roles (energy, material, or signal).  Many of the models in the repository were developed to follow this form, or some variation of it, however, more complex modeling use-cases have led us to expand our conception of what can/should be modeled with a function or flow. More generally, 

- Flows define *shared data structures*, meaning interacting variables and
- Functions define *behaviors*, meaning things to be done to flows.

These functions and flows are connected via containment relationships in an undirected graph, meaning that functions can be run in any order within a time-step to enable faults to propagate throughout the model graph. This is a very general representation, but also leads to pit-falls if the model is too complex, since this behavior needs to be convergent within each timestep. The following gives some advice for conventions to follow in models based on their size/scope.

**Small Models**

Small models have a few functions with simple behaviors that are being loaded in simple ways. A good example of this is the [Pump Example](../examples/water_pump/demo_fault_analysis.ipynb) and [EPS Example](../examples/electric_power_system/demo_static_models.ipynb) , where the model is a simple translation of inputs (defined in input functions) to outputs (defined in output functions). These models have the most ability to follow the [Functional Basis modeling ontology](https://link.springer.com/article/10.1007/s00163-001-0008-3>) (with `import_x` being inputs and `output_x` being outputs of the system), as well as use static_behavior methods. It is also possible to model many different modes with full behavioral detail, since the system itself is not too complicated. Technical debt and development process is less of a consideration in these models, but should still not be ignored. A typical structure for a model would be:

* Architecture
	* flows
		* X
		* Y
	* functions
		* Import_X
		* Change_X_to_Y
		* Export_Y

**System Models**

Moderate-size system models are models which have a control/planning system (e.g., something that tells it what to do at any given time). They also often interact with their environment in complex ways. A typical structure for a model would be: 

* Architecture
	* flows
		* Environment, Location, etc 		(place the system is located in and its place in it)
		* Power, Actions, etc				(internal power/other physical states)
		* Commands,Communications, etc 	(external commands/comms with an operator)
	* functions
		* Affect_Environment 				(Physical behaviors the system performs on the environment)
		* Control_System 					(Controls, Planning, Perception, etc)
		* Distribute_Energy, Hold_X, etc 	(Internal components, etc)

A good example of this are the Drone and Rover models. Models like this are simply more complex and thus require more care and attention to avoid the accumulation of technical debt. It may be desirable for some of the more complex functions to be specified tested in isolation, and developed in their own files. Finally, flows such as `Environment` may require developing custom visualization methods (maps, etc) to show the how the system interacts with its environment.


**System of Systems Models**

Systems of Systems models involve the interaction of multiple systems in a single model. These models are much more complex and thus require very good development practices to develop to maturity. A typical structure for a model for this might be:

* Architecture
	* flows
		* Environment						(place the systems are located in)
		* Location(s)						(individual states of the agents)
		* Communication(s) 				(agent interactions with each other
	* functions
		* Asset/Agent(s)					(individual system models)
		* AgentController(s)				(coordinator which issues commands to each system)

Note that, unlike other model types, System of Systems models very often will have multiple copies of functions and flows instantiated in the model. As a result, it is important to use dedicated model structures to the overall structure from being intractable. Specifically, multiple copies of flows can be handled using the `MultiFlow` class while Communications between agents can be handled using the `CommsFlow` class. The `ModelTypeGraph` graph representation can be used to represent the model as just the types involved (rather than all instantiations). In general, it can be helpful to create tests/analyses for individual agents in addition to the overall system.

## Structuring your Project Repository

When you use fmdtools for your own project, this code will invariably end up in a folder. 

Best practice is to structure this folder using modern Python software development tools and standards (e.g., uv, git, pyproject.toml, pytest, etc). Meaning:
1. Set up your folder as a git repository so you have version control.
2. Use uv to manage your project, its dependencies, and set up a virtual environment for your project (ideally independent of others)
3. Set up your repository using a git editable install (e.g., `uv pip install -e .`) to ensure consistency between your installation and builds
4. Document project dependencies and usage in the README and pyproject.toml

When it comes to structuring folders for various simulations, we have adopted the following conventions (see `/examples` in the repository):

* Models should be named `model_variantname.py`, where variantname is the variant in the file. If there is a main model, name it `model_main.py`.
* Tests should be named `test_testname.py` (standard python conventions).
* Scripts should be named `script_scriptname.py`. 
* Notebooks should be named `notebooktype_notebookname.ipynb`, where `notebooktype` is the type of notebook (e.g., `paper` for generating figures for a paper or `demo` for showcasing a project)
* Any and all outputs from these files should be placed in folders with the name `outputs_generating_file_name/`

These conventions help keep repositories from getting too cluttered and ensure that you can quickly find the file you need and understand its need and purpose.

If your project gets big, or you have multiple models with shared classes, you may consider building a library out of it. In this case `examples/airspacelib` or the fmdtools library itself may be a good example of how to structure the folder.

```
src/libraryname/ 	# folder for library packages
examples/ 			# folder for examples or usages of the libary
tests/ 				# folder for tests and pytest configuration
docs/ 				# folder for documentation
pyproject.toml 		# Overall library configuration for tools
README.md 			# Readme and overview of the library
```

In this case, the contents of examples would apply the naming schema above, while the library itself would have shorter names (e.g., names of their contained class(es)).

## Style advice

Development of fmdtools models should follow the [PEP 8 Style Guide](https://peps.python.org/pep-0008/#introduction) as much as possible. While this won't be entirely re-iterated here, the following applies:

* Use CamelCase for classes like `fmdtools.define.architecture.function.FunctionArchitecture`, `fmdtools.define.block.function.Function`, `fmdtools.define.flow.base.Flow`, etc. Use lowercase for object instantiations of these classes, and lower_case_with_underscores (e.g. do_this()) for methods/functions.
	* if a model class is named Model (e.g., Drone), the instance can be named model_X, where X is an identifying string for the model being used (e.g. drone_test). 
* Names should be descriptive, but keep the length down. Use abbreviations if needed.
* Try to use the code formatting structure to show what your code is doing as much as possible. Single-line if statements can be good for this, as long as they don't go too long.
* Python one-liners can be fun, but try to keep them short enough to be able to read. 
* If a file is >1000 lines, you may want to split it into multiple files, for the model, complex classes, visualization, analysis, tests, etc.
* fmdtools lets you specify names for functions/flows. Keep these consistent with the class names but consider making them short to enable visualization on model graphs and throughout the code.
* It's `fmdtools`. Not `Fmdtools` or `fmd tool`. Even when it starts the sentence.

See also
--------

* [PEP 8 Style Guide](https://peps.python.org/pep-0008/#introduction)
* [Technical debt](https://en.wikipedia.org/wiki/Technical_debt)
* [Code smell](https://en.wikipedia.org/wiki/Code_smell)
* [Anti-patterns](https://en.wikipedia.org/wiki/Anti-pattern)
* [Iterative development](https://en.wikipedia.org/wiki/Iterative_and_incremental_development)
* [Python Programming Idioms](https://en.wikibooks.org/wiki/Python_Programming/Idioms)
* [The Zen of Python](https://en.wikipedia.org/wiki/Zen_of_Python)

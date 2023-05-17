Development Guide
===========================

How to Write Good Models - Development Best Practices
--------------------------------

Plan your model to avoid technical debt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simple, small models are relatively easy to define in fmdtools with a few functions, flows, and behaviors. As such, it can be easy to get in the habit of not planning or organizing development in a systematic, which leads to issues when developing larger models: specifically, code that is *written into existence* instead of designed, planned, edited, tested, and documented. This leads to `Technical debt <https://en.wikipedia.org/wiki/Technical_debt/>`_, which is the inherent difficulty of modifying code that was written ad-hoc rather than designed. Unless this technical debt is resolved, the ability to modify a model (e.g., to add new behaviors, conduct analyses, etc) will be limited by the complicated and unwieldy existing code. 

The next subsections give some advice to help avoid technical debt, based on lessons learned developing fmdtools models over the past few years.

Don't copy, inherit and functionalize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Copy-and-paste can be a useful concept, but often gets over-relied upon by novice model developers who want to create several variants of the same programming structure. However, in the world of systems engineering (and software development), there are many cases where developers should be using `class inheritance <https://www.w3schools.com/python/python_inheritance.asp/>`_ and `writing functions <https://swcarpentry.github.io/python-novice-gapminder/16-writing-functions/index.html/>`_ instead. 

The advantages of inheritance are: 

#. it reduces the bulk amount of code required to represent functions, making code more comprehensible
#. it makes the distinction between similar classes easier to distinguish, since *there is no redundant code*
#. it makes it easier to edit code afterward, since the developer *only has to edit in one place* for it to apply to all the relevant types
#. it makes testing easier, since common methods only need to be tested once

In fmdtools, these patterns can be helpful:

* Instead of creating two very similar :class:`fmdtools.define.FxnBlock` classes (e.g. Drone and PilotedAircraft) and copying code between each, create a single class (e.g. Aircraft) with common methods/structures (e.g., Fly, Taxi, Park, etc.) and then use sub-classes to extend and/or replace methods/structures the common class as needed (e.g., Autonomous Navigation in the Drone vs. Piloted Navigation in the normal aircraft).
* In a :class:`fmdtools.define.Model` definition, instead of having several flow dictionaries with similar/the same data inputs, define one dictionary at the beginning and use it in each of the :meth:`fmdtools.define.Model.add_flow` methods.
* Instead copying code for the same operations to several different places in a model, write a single function instead. This method can then be documented/tested and extended to a variety of different use-cases which require the same basic operation to be done. 

This is an incomplete list. In general, it can be a helpful limitation to *try to avoid using copy-and-paste as much as possible.* Instead if a piece of code needs to be run more than once in more than once place, write a function or method which will be used everywhere. The idea should be to *write the code once, and run it everywhere.*

Document your code, sometimes *before* your write it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general, Python coding style aspires to be `as self-documenting <https://en.wikipedia.org/wiki/Self-documenting_code/>`_ as possible. However, this is not a replacement for documentation. In general, novice developers think of documentation as something which happens at the end of the software development process, as something to primarily assist users. 

This neglects the major benefits of documentation in the development process. Specifically:

#. It helps other people understand how to use and *contribute to* your code,
#. It helps define what your code is supposed to do, its interfaces, and the desired behavior, and
#. As a result, it helps you understand your code.

For fmdtools models, documentation should at the very least take the following form:

* A README (in markdown or rst) that explains how to set up the model environment (e.g., requirements/dependencies), as well as explains the structure of the folder/repo (model file, tests, analyses, etc.)
* Documented examples of using the code. Usually you can use a jupyter notebook for this to show the different analyses you can run with your model.
* Docstrings which document the classes and functions which make up your model. These are most important for development and should include:
	* An overall module description (top of file)
	* Docstrings for flows: What does the state represent? What are the states? What values may these take?
	* Docstrings for :class:`fmdtools.define.FxnBlock`: What are the states, parameters, behaviors, and modes?
	* For any method/function, try to follow existing docstring conventions, with a summary of the purpose/behavior of the method, and a description of all input/output data types.

Documentation can best be thought of as a *contract that your code should fulfill*. As such, it can be very helpful to think of the documentation first, as a way of specifying your work. Tests (formal and informal) can then be defined based on the stated behavior of the function. It is thus recommended to *document your code as you write it*, instead of waiting until the end of the development process, to avoid technical debt. 

Don't get ahead of yourself--try to get a running simulation first
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the model development process, it can often be tempting to try to model every single mode or behavior in immense detail from the get-go. This is motivated by a desire to acheive realism, but can lead to issues from a project management and integration perspective. A model does not have much meaning outside a simulation or analysis, and, as such, development needs to be motivated *first* by getting a working simulation and *then* by adding detail. These simulations are the key feedback loop for determining whether model code is embodying desired behavior. 

A very basic model development process should thus proceed:

#. Create model file and create place-holder major function/flow classes
#. Connect classes in a model file and visualize structure
#. Create low-fidelity model behaviors and verify in nominal scenario
#. Add hazard metrics in `find_classification` 
#. Add more detailed behaviors (e.g., modes, actions, components, etc) as needed
#. Perform more complex analyses...

In general, it is bad to spend a lot of time developing a model without running any sort of simulation for verification purposes. This toolkit has been designed to enable the use of simulations early in the development process, and it is best to use these features earlier rather than later.

Finally, *smaller, incremental iterations are better than large iterations.* Instead of spending time implementing large sections of code at once (with documentation and testing TBD), instead implement small sections of code that you can then document, test, and edit immediately after. Using these small iterative cycles can increase code quality by ensuring that large blocks of undocumented/untested (and ultimately unreliable) code don't make it into your project, only for you to have to deal with it later.

Preserve your prototype setup by formalizing it as a test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Testing code is something which is often neglected in the development process, as something to do when the project is finished (i.e., as an assurance rather than development task). Simultaneously, developers often iterate over temporary scripts and code snippets during development to ensure that it works as expected in what is essentially an informal testing process. The major problem with this process is that these tests are easily lost and are only run one at a time, making it difficult to verify that code works after it has been modified.

Instead, it is best to *formalize scripts into tests*. This can be done with Python's `unittest <https://docs.python.org/3/library/unittest.html>`_ module, which integrates well with existing python IDEs and enables execution of several different tests in a sequence. Instead of losing prototype code, one can easily place this code into a `test_X` method and use it iteratively in the development process to ensure that the code still works as intended. This is true even for more "qualitative" prototype script, where the output that is being iterated over is a plot of results. Rather than abandoning a prototyping setup like this, (e.g., by commenting it out), a much better approach is to formalize the script as a test which can be run at the will of the user when desired. In this case, the plot should show the analysis and describe expected results so that it can be quickly verified. The testing of plots is enabled with the function :func:`fmdtools.analyze.plot.suite_for_plots`, which enables you to filter plotting tests out of a model's test suite (or specify only running specific tests/showing specific plots). 

While testing is an assurance activity, it should also be considered a development activity. Testing ensures that the changes made to code don't cause it to take on undesired behaviors, or be unable to operate with its interfacing functions. To enable tests to continue to be useful through the modelling process, they should be given meaningful names as well as descriptions describing what is being tested by the test (and why).

Finally, don't create tests solely to create tests. Tests should have a specific purpose in mind ideally single tests should cover as many considerations as possible, rather than creating new tests for each individual consideration. As in model development, try to avoid bloat as much as possible. If the desire is to cover every edge-case, try to parameterize tests over these cases instead of creating individual test methods.

Edit your code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nature of writing code is a messy process--often we spend a considerable amount of time getting code to a place where it "works" (i.e., runs) and leave it as-is. The problem with doing this over and over is that it neglects the syntax, documetation, and structural aspects of coding and thus contributes to technical debt. One of the best ways to avoid this from impacting development too much is to edit code after writing it.

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

Structuring a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fmdtools was originally developed around a very simple use-case of modelling physical behaviors using a Function/Flow ontology, where Functions (referred to as "technical functions") are supposed to be the high-level roles to be performed in the system, while flows are the data passed between these roles (energy, material, or signal).  Many of the models in the repository were developed to follow this form, or some variation on it, however, more complex modelling use-cases have led us to need to expand our conception of what can/should be modelled with a function or flow. More generally, 
- Flows define *shared data structures*, meaning interacting variables
- Functions define *behaviors*, meaning things to be done to flows.

These functions and flows are connected via containment relationships in an undirected graph, meaning that functions can be run in any order within a time-step to enable faults to propogate throughout the model graph. This is a very general representation, but also leads to pit-falls if the model is too complex, since this behavior needs to be convergent within each timestep. The following gives some advice for conventions to follow in models based on their size/scope.

**Small Models**

Small models have a few functions with simple behaviors that are being loaded in simple ways. A good example of this is the `Pump Example <example_pump/Pump_Example_Notebook.ipynb>`_ and `EPS Example <example_eps/EPS_Example_Notebook.ipynb>`_ , where the model is a simple translation of inputs to (defined in input functions) outputs (defined in output functions. These models have the most ability to follow the functional basis modelling ontology (with `import_x` loadings and `output_x` outputs) , as well as use static_behavior methods. It is also possible to model many different modes with full behavioral detail, since the system itself is not too complicated. Technical debt and development process is less of a consideration in these models, but should still not be ignored. A typical structure for a model would be:

* Model
	*flows
		* X
		* Y
	*functions
		* Import_X
		* Change_X_to_Y
		* Export_Y

**System Models**

Moderate-size system models are models which have a control/planning system (e.g., something that tells it what to do at any given time). They also often interact with their environment in complex ways. A typical structure for a model would be: 

* Model
	*flows
		* Environment, Location, etc 		(place the system is located in and its place in it)
		* Power, Actions, etc				(internal power/other physical states)
		* Commands,Communications, etc 	(external commands/comms with an operator)
	*functions
		* Affect_Environment 				(Physical behaviors the system performs on the environment)
		* Control_System 					(Controls, Planning, Perception, etc)
		* Distribute_Energy, Hold_X, etc 	(Internal components, etc)

A good example of this are the Drone and Rover models. Models like this are simply more complex and thus require more care and attention to avoid the accumulation of technical debt. It may be desireable for some of the more complex functions to be specified tested in isolation, and developed in their own files. Finally, flows such as `Environment` may require developing custom visualization methods (maps, etc) to show the how the system interacts with its environment.


**System of Systems Models**

Systems of Systems models involve the interaction of multiple systems in a single model. These models are much more complex and thus require very good development practices to develop to maturity. A typical structure for a model for this might be:

* Model
	*flows
		* Environment						(place the systems are located in)
		* Location(s)						(individual states of the agents)
		* Communication(s) 				(agent interactions with each other
	*functions
		* Asset/Agent(s)					(individual system models)
		* AgentController(s)				(coordinator which issues commans to each system)

Note that, unlike other model types, System of Systems models very often will have multiple copies of functions and flows instantiated in the model. As a result, it is important to use dedicated model structures to the overall structure from being intractible. Specifically multiple copies of flows can be handled using the `MultiFlow` class while Communications between agents can be handled using the `CommsFlow` class. The `ModelTypeGraph` graph representation can be used to represent the model as just the types involved (rather than all instantiations). In general, it can be helpful to create tests/analyses for individual agents in addition to the overall system.

Use model constructs to simplify your code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The fmdtools codebase is quite large, and, as a result, it can be tempting to dive into modelling before learning about all of its capabilities. The problem with this is that many of these capabilities and interfaces are there to make your life easier, provided you understand and use them correctly. Below are some commonly-misunderstood constructs to integrate into your code:

* :class:`fmdtools.define.Common` has a number of very basic operations which can be used in all model structures to reduce the length of lines dedicated solely to assignment and passing variables between constructs. Using these methods can furthermore enable one to more simply perform vector operations with reduced syntax.
* :class:`fmdtools.define.Timer` can be used very simply using to represent timed behavior and state-transitions 
* While modes can be used to describe fault modes in a very general way, faulty behavior that can also be queried from the model using the concept of a *disturbance*, which is merely a change in a given variable value. While disturbances are less general, they requie much less to be implemented in the model.
* parameters and parameter-generating functions are helpful for understanding the model operating envelope. In general, try to avoid having parameters that duplicate each other in some way.
* Randomness can be used throughout, but use the specified interfaces (`seed`, :meth:`fmdtools.define.Block.assoc_rand_states`, etc.) so that a single seed is used to generate all of the rngs in the model. Not using these interfaces can get you in trouble.
* A variety of custom attributes can be added to :class:`fmdtools.define.FxnBlock` and :class:`fmdtools.define.Flow`, but not every custom attribute is going to work with staged execution and parallelism options. In general, all function constructs that change should be `states`, while parameters may be attributes. If you want to do something fancy with data structures, you may need to re-implement :class:`fmdtools.define.FxnBlock` methods for copying and returning states to `propagate`.
* If there's something that you'd like to do in an fmdtools model that is difficult with existing model structures, consider filing a bug report before implementing you own ad-hoc solution. Alternatively, try devoping your solution as a *feature* rather than a hack to solve a single use-case. If the features is in our scope and well-developed, we may try to incorporate it in our next release.

Style advice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Development of fmdtools models should follow the `PEP 8 Style Guide <https://peps.python.org/pep-0008/#introduction>`_ as much as possible. While this won't be entirely re-iterated here, the following applies:

* Use CamelCase for classes like :class:`fmdtools.define.Model`, :class:`fmdtools.define.FxnBlock`, :class:`fmdtools.define.Flow`, :class:`fmdtools.define.Component`, etc. Use lowercase for object instantiations of these classes, and lower_case_with_underscores (e.g. do_this()) for methods/functions.
	* if a model class is named Model (e.g., Drone), the instance should be named model_X, where X is an identifying string for the model being used (e.g. drone_test). 
* Names should be descriptive, but keep the length down. Use abbreviations if needed.
* Try to use the code formatting structure to show what your code is doing as much as possible. Single-line if statements can be good for this, as long as they don't go too long.
* Python one-liners can be fun, but try to keep them short enough to be able to read. 
* If a file is >1000 lines, you may want to split it into multiple files, for the model, complex classes, visualization, analysis, tests, etc.
* fmdtools lets you specify names for functions/flows. Keep these consistent with with the class names but consider making them short to enable visualization on model graphs and throughout the code.
* It's `fmdtools`. Not `Fmdtools` or `fmd tool`. Even when it starts the sentence.

See also
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `PEP 8 Style Guide <https://peps.python.org/pep-0008/#introduction>`_
* `Technical debt <https://en.wikipedia.org/wiki/Technical_debt/>`_
* `Code smell <https://en.wikipedia.org/wiki/Code_smell>`_
* `Anti-patterns <https://en.wikipedia.org/wiki/Anti-pattern>`_
* `Iterative development <https://en.wikipedia.org/wiki/Iterative_and_incremental_development>`_
* `Python Programming Idioms <https://en.wikibooks.org/wiki/Python_Programming/Idioms>`_
* `The Zen of Python <https://en.wikipedia.org/wiki/Zen_of_Python>`_
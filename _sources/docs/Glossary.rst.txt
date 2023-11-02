Glossary
==============================================

.. glossary::

	Function
		A piece of functionality in a system which has its own defined behavior, modes, and flow connections, and may be further instantiated by a :term:`component architecture` or :term:`action sequence graph`. In general, functions are the main major building block of a model defining how the different pieces of the system behave. Functions in fmdtools are specified by extending the :class:`fmdtools.define.blockFxnBlock` class.
		
	Flow
		A data structure which connects functions--traditionally energy, material, or signal. Defined using the :class:`fmdtools.define.flow.Flow` class.
	
	Role
		A defined attribute of an fmdtools class which refers to a user-defined (or default) subclass of a corresponding fmdtools data structure. For example, Blocks have the role `Block.s` (for state) which may be filled by a subclass of :class:`fmdtools.define.state.State`.

	Internal Flow
		A flow object that is internal to a :class:`fmdtools.define.block.FxnBlock` which is not present in the overall model definition.
	
	Model
		A simulation that defines system behavior. Models contain functions and flows, their graph connections, parameters related to the simulation configuration, as well as methods for classifying simulations. Models are specified using the :class:`fmdtools.define.model.Model`: class.
	
	Behavior
		How the states of a system unfold over time, including in the various :term:`mode` s it may encounter. Defined in :term:`Function` s, :term:`Component` s, and :term:`Action` s using :meth:`fmdtools.define.Block.behavior`, :meth:`fmdtools.define.Block.static_behavior`, :meth:`fmdtools.define.Block.dynamic_behavior`, and :meth:`fmdtools.define.Block.condfaults`
	
	Graph
		A view of simulation construct connections and/or relationships embodied by the :class:`fmdtools.analyze.graph.Graph` class and sub-classes (which uses networkx to represent the structure itself).
	
	Component
		A physical that embodies specific behavior for a :term:`function`. May have :term:`mode` s and :term:`behavior` s of its own. Specified by extending the :class:`fmdtools.define.block.Component` class.
		
	Component Architecture
		The physical embodiment of a :term:`function` that encompasses multiple :term:`Component` s. Represented via the :class:`fmdtools.define.block.CompArch` class. 
	
	Mode
		Discrete modifications of a :term:`behavior` specified as entries in the :meth:`fmdtools.define.mode.Mode` class. Often used to control if/else statements in a :meth:`fmdtools.define.Block.behavior` method.
	
	Fault Mode
		Undesired :term:`mode`, which leads to hazardous behavior. For example, a lamp may have "burn-out" and "flicker" modes
	
	Operational Mode
		Defined :term:`mode` that the system progresses through as a part of its desired functioning. For example, a light switch may be in "on" and "off" modes.

	Action Sequence Graph
		An instance of the :class:`fmdtools.define.block.ASG` which embodies a (human or autonomous) :term:`agent` 's sequence of tasks which it performs to accomplish a certain function. 
	
	Agent
		An actor which controls behaviors in a system. May be modelled as a :term:`function`.
		
	Environment
		The uncontrolled aspect of a system which may effect system inputs and behaviors. May be modelled as a :term:`function`.
	
	Action
		A specific task to be performed by an :term:`agent` used to represent human/autonomous operations. May be specified by extending the :class:`fmdtools.define.block.Action` class and added to a :class:`fmdtools.define.block.FxnBlock` as a part of an Action Sequence Graph :class:`fmdtools.define.block.ASG`
	
	Rate
		The expected occurence (frequency) of a given :term:`mode`, which may be specified in a number of ways using :meth:`fmdtools.define.mode.Mode.faultmodes`.
		
	Cost
		A metric used to define severity of a scenario. While cost is defined in a monetary sense, it should often be defined holistically to account for indirect costs and externalities (e.g., safety, disruption, etc). One of the default outputs from :meth:`fmdtools.define.block.Simulable.find_classification` for models or blocks.
		
	Expected Cost
		A metric used to define risk of a scenario, calculated my multiplying the :term:`rate` and :term:`cost`.
		
	Endclass
		The end-state classification given from :meth:`fmdtools.define.block.Simulable.find_classification`.
	
	Scenario
		A specific set of inputs to a simulation, including :class:`fmdtools.define.model.Model` parameters, :term:`Fault Mode` s, and :term:`Disturbances`. Defined in :class:`fmdtools.sim.scenario.Scenario`.
		
	Disturbances
		A specific sequence of variable values over time which may modify system behavior.

	Sample
		A set of :term:`scenario` s to simulate a model over to represent certain hazards or parameters of interest. May be generated using :class:`fmdtools.sim.sample.FaultSample` for fault modes or :class:`fmdtools.sim.sample.ParameterSample` for nominal parameters. 
	
	Nested Approach
		The result of simulating a fault sampling :term:`Approach` (:class:`fmdtools.sim.sample.SampleApproach`) within a nominal :term:`Approach` (:class:`fmdtools.sim.sample.ParameterSample`). Created in :func:`fmdtools.sim.propagate.nested_approach`
	
	Static Propagation
		The undirected propagation of model behaviors within a timestep. Defined for each function using :meth:`fmdtools.define.block.FxnBlock.static_behavior`, which may run multiple times in a timestep until behavior has converged.
	
	Dynamic Propagation
		The progression of model states over time. Defined for each function using :meth:`fmdtools.define.block.FxnBlock.dynamic_behavior`, which runs once per timestep.
	
	Propagation
		The simulation of :class:`fmdtools.define.model.Model` :term:`behavior` s, including the passing of :term:`flow` s between :term:`function` s and the progression of model states over time.
	
	Resilience
		The expectation of defined performance metrics over time over a set of hazardous :term:`scenario` s, often defined in terms of the deviation from their nominal values.
	
	End-state
		The state of a :class:`fmdtools.define.block.Simulable` at the final time-step of a simulation.
	
	FMEA
		A table outlining the risks of hazardous :term:`scenario` s in terms of their rate, severity, and expected risk. By default, the :mod:`fmdtools.analyze.tabulate` module produces cost-based FMEAS, with the metrics of interest being :term:`rate`, :term:`cost`, and :term:`expected cost`, however these functions can be tailored to the metrics of interest.
	
	Behavior Over Time
		How a the states of a system unfold over time. Defined using :term:`behavior`.
	
	Model History
		A history of model states over a set of timesteps defined in :class:`fmdtools.analyze.history.History`. Returned in fmdtools as a nested dictionary from methods in :mod:`fmdtools.sim.propagate`.

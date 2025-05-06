FRDL Version 0.7
================


Overview
^^^^^^^^

FRDL stands for the Functional Reasoning Design Language, which is a graphical language for describing the functional architecture of a system. A functional architecture (often known as a `functional model <https://en.wikipedia.org/wiki/Function_model>`_, `functional decomposition <https://en.wikipedia.org/wiki/Functional_decomposition>`_, `function-flow block diagram <https://en.wikipedia.org/wiki/Functional_flow_block_diagram>`_)  describes the high-level functionalities (or, functions) that are to be embodied by a system, along with the interactions between these functionalities. Diagrams of functional architectures (like those enabled by FRDL) enable *functional reasoning* by providing a means to abstract away irrelevent aspects of system structure and behavior while focusing on the key (that is, functional) elements. Functional reasoning can support a range of design and analysis activities by encouraging clear reasoning about how a system operates and what it is supposed to do at a high level. This can facilitate many important design and analysis activities, including `product design (and especially redesign) <https://link.springer.com/book/10.1007/978-1-84628-319-2>`_, behavioral simulation, and `hazard analysis <https://en.wikipedia.org/wiki/Failure_mode,_effects,_and_criticality_analysis>`_.

What can you use FRDL for?
--------------------------

As a language, FRDL can be used to support a range of activities where functional reasoning can be helpful, and is thus meant as a drop-in replacement to many existing functional modelling languages (such as block diagrams, energy-materials-signals models, etc.). However, the main focus of FRDL has been improving hazard analysis through its explicit representation of behavioral interactions. In the context of hazard analysis, FRDL models can be used to identify potential causes and effects of hazardous conditions, by tracing the interactions between functions that could induce or be affected by these conditions.

Why FRDL?
---------

FRDL was developed to improve the analytical rigor of the diagrams that analysts typically use to perform hazard analysis to better support causal reasoning. Compared to other languages, it has a few key features and advantages:

- FRDL encourages functional reasoning, which is a key and long-appreciated part of hazard analysis. While existing hazard analysis approaches reccomend using a ``functional'' approache early in design, they often do not provide well-defined idea of what a "function" is, which can lead to confusion and ultimately poor reasoning. FRDL resolves this issue by providing a *way of thinking* about function which comports both to 

- FRDL encourages (or rather, requires) the consideration of behavioral interactions. These behavioral interactions are key for understanding both (1) how hazardous conditions can lead to downsteam hazards and (2) how hazarous conditions can arise in a system. Without this explicit representation of behavioral interactions, analysts are left to either (1) use very simplistic reasoning based on what models do represent (e.g., if a function within another function fails, the containing function must also fail) or (2) rely on their own mental models of system behavior (which at the very least lack traceability and justification, and at the very most could be undetectably flawed).

- FRDL offers a scalable representation of system complexity that enables the representation of diverse interactions in highly complex systems. Unlike other methods--such as STAMP and block diagrams, which encode interactions as arrows, these interactions are represented in FRDL as nodes, making it possible to more scalably represent diverse interactions in a system that affect more than one function. This approach also encourages the grouping of like interactions, as well as the representation of wide-reaching interactions such as communications in a system-of-systems.

- FRDL explicitly supports annotations to support more detailed information to be used in analysis when it is available. Specifically, FRDL annotations can be used to better represent the dynamics of behavior (e.g., when functionality is active versus inactive) to inform the analysis of system resilience to hazards.

- FRDL enables the direct integration with the fmdtools simulation library, meaning that hazard models in FRDL can be directly implemented in simulation (rather than what is typical--simulations of system behavior looking much different from the original functional model).


Relationship with fmdtools
--------------------------

FRDL has been developed as a specification for fmdtools models, such that a functional architecture developed in FRDL can be directly implemented as an fmdtools :class:`FunctionArchitecture`. While this correspondence is supported by the fmdtools data model, the display of these structures often veers from the FRDL specification (see :ref:`fmdtools_graph_style`) due to limitations in the graphical libraries fmdtools uses to display architectures. Additionally, fmdtools does not currently support defining/displaying the annotations in frdl, showing only the graph structures instead. One of the development goals for fmdtools is thus to bring these features of FRDL into the library, as well as provide an interface for translating externally-supplied FRDL models into fmdtools Architecture classes.

Specification
^^^^^^^^^^^^^

FRDL represents System `Architectures`_ by composing node and edge `Primitives`_ into an overall graph of system interactions with a defined `Interpretation`_.

Primitives
----------

Primitives are the nodes and edges used to create `Architectures`_, including:

Blocks
""""""

.. figure:: figures/frdl/primitives/blocks.svg
   :width: 800
   :alt: FRDL Block Classes

Blocks in FRDL are behavioral elements of the system, meaning they are expected to perform a given behavior. Behavior is any operation, such as an equation, modification, or constraint that the block imposes on the system. For example, ``flying,`` ``transfering heat,`` and ``set x to one`` may all be considered behaviors which blocks would embody. Blocks may further be annotated by `Block Annotations`_ to describe structural and behavioral properties of the block. FRDL supports three major block classes, shown above and described below.

Components
''''''''''

Components refer to concrete parts that physically make up a system. These components may have multiple behaviors that interact in different ways. For example, an engine, a wheel, and a brake would all be different components of a car. Components may be used in the context of `Component Architectures`_ to represent the embodyment of a function in real-world parts. Components correspond directly to the fmdtools :class:`~fmdtools.define.block.component.Component` class.

Actions
'''''''

Actions refer to discrete, logical steps preformed in controlling a system to accomplish an overall task. For example, "Turn On"/"Turn Off" and "Turn left"/"Turn right" are both actions that a user might perform on the system in order to use it. Actions are logical behaviors, and thus should be thought of as software (e.g., control logic) or human reasoning. Actions correspond directly to the fmdtools :class:`~fmdtools.define.block.action.Action` class.

Functions
'''''''''

Functions describe *generic abstract functionality* that the system is to be embody. As opposed to what a system ``is'' (e.g., a collection of parts), functions describe what a system ``does.'' Functions may be labeled as verbs acting on nouns (e.g., "process ore into iron"), verbs (e.g., "navigate"), or a set of tasks (e.g., "store and supply energy"). As primarily behavioral elements, functions can additionally be thought of as comprising one or multiple governing equations of the system.  Functions correspond directly to the fmdtools :class:`~fmdtools.define.block.function.Function` class.

Because functions are hybrid elements, they can be embodied by physical components (see: `Function/Action Relationship`_), logical actions (see: `Function/Component Relationship`_), more elemental functions, and architectures.

Block Annotations
"""""""""""""""""

.. figure:: figures/frdl/primitives/block_annotations
   :width: 800
   :alt: FRDL Block Annotations

Annotations may be used to clarify known properties of the block. An overview of these annotations (described next) is provided above. While it is not required to use any of these annotations, they are provided as a part of the language to better inform analyses with relevant information.

Dynamics Tag
''''''''''''

From left to right, dynamics tags specify:

- the start time or condition, which may be "i" to specify that it starts when activated (if activations are numbered their ID may be provided here also), "s" to specify that it starts at scenario start, and a value (e.g., "10s") to specify a given time.
- the change interval, which may be "dt" if unspecified/continuous or a given value (e.g., "5s") if there is a given timestep for the behavior
- the end time or condition, which may be "o" to specify that it ends at a given activation propagating out (if activations are numbered an ID may be provided here), "e" to specify that it ends at the end of the scenario, and a number (e.g., "120s") to specify a given end time.

An additional arrow symbol a the upper left of the tag specifies that the block starts the scenario, while a fork symbol at the upper right of the tag specifies that the block ends the scenario.

Dynamics tags are placed in the upper left corner of the block. 

Behavior Type Annotation
''''''''''''''''''''''''

Behavior type annotations help provide more details at the block level which may help describe the block. Behavior type annotations are especially relevant to Functions, where they can help explain the expected embodiment of the function (e.g. "Battery" for a "Supply Power" function) and/or what is expected from the function. These annotations may be placed immediately below the name of the block.

Architecture Tag
''''''''''''''''

Architecture tags specify whether a block contains within itself a Component, Action, or Function architecture, or some combination, by listing each letter (C, A, and/or F) in the tag. These are primarily to be used when there are additional architecture diagrams which may be referenced to describe the block. The architectue tag is placed in the lower-left corner of the block.

Ontology Tag
''''''''''''

The ontology tag refers to the scope the block takes and its role within that scope. The scopes include:

- Internal, meaning that the block is a part of the designed/operated system. Within this scope, the "required" role specifies that the block is a user/operator requirement, the "supporting" role specifies that the block supports a required block, while the "constraining" role specifies that the block is a technical constraint placed on the system  (e.g., laws of physics that must be followed regardless of whether they are desired).
- External, meaning that the block is external to the system. Within this scope, the "defined" role specifies that the block is a well-defined interface and not likely to vary, the "variable" role specifies that the block may vary outside of the control of the user/operator, and the "adversarial" role specifies that the block is taking actions against the interest/functioning of the system.

The ontology tag is placed in the lower right of the block.

Flows
"""""

Flows represent the means by which blocks may interact, and may be thought of as shared variables, inputs/outputs, or a shared environment. For example, in a circuit, electricity represents the flow between elements of the circuit.

.. figure:: figures/frdl/primitives/types_of_flow.svg
   :width: 800
   :alt: FRDL Flow classes.

There are three main types of flow, shown above. Base flows represent directly coupled links between functions (i.e., an aggregation), meaning that properties of the flow in one block directly correspond to the properties of the flow in another. Flows correspond directly to the :class:`~fmdtools.define.flow.base.Flow` class in fmdtools.

Flows have additional variations, including `MultiFlow`_ and `CommsFlow`_s, described next.

MultiFlow
'''''''''

MultiFlows are flows with some level of multiplicity, which may be used when blocks *may* have their own individual "views" of the flow. One block's MultiFlow properties thus may not necessarily correspond to the properties of another block's MultiFlow. MultiFlows correspond directly to the :class:`~fmdtools.define.flow.multiflow.MultiFlow` class in fmdtools.

CommsFlow
'''''''''

CommsFlows are flows that make up a communications network (or mesh) between different blocks. CommsFlows thus specify a given structure for sending/recieving flow properties to any or all other connected blocks. CommsFlows correspond directly to the :class:`~fmdtools.define.flow.commsflow.CommsFlow` class in fmdtools.

Relationships
"""""""""""""

Relationships are edges connecting nodes in a model graph that specify how nodes (blocks and flows) relate to each other. When used in larger UML or SysML ecosystem, edges can specify a wide range of logical and behavioral concepts. For the purpose of architecture modelling, FRDL relies on connection, activation, and propagation edges (shown below) to specify behavioral interactions.


Connection
''''''''''

.. figure:: figures/frdl/primitives/flowconnection.svg
   :alt: Connection

Connection arrows (shown above) specify that a flow is to be considered jointly a part of two or more blocks. In doing so, the flow ``connects'' the blocks. This connection is equivalent to a **shared association** in UML/SysML, meaning that the flow is considered a part of all blocks it is connected to, but is not owned by any of them.

While connection arrows specify this joint connection, they do not in and of themselves specify a behavioral interaction between connected blocks. Instead they specify that a given flow is a *means* by which the blocks could interact.

Activation
''''''''''

.. figure:: figures/frdl/primitives/activation.svg
   :alt: Activation

Activation arrows (shown above) specify that a condition in one block causes a condition in (or, ``activates'') another block. Activation arrows can be annotated with text specifying the condition that causes the activation. Activation arrows can be used in Action Architectures to specify sequences of tasks that complete one after the other (similar to a activity diagram or finite state machine).


Propagation
'''''''''''

.. figure:: figures/frdl/primitives/propagation.svg
   :alt: Propagation

Propagation arrows (shown above) represent the propagation of behavior between blocks via flows. In this way propagation arrows represent the composition of Connection and Activation relationships, specifying both (1) the means by which blocks interact and (2) the specific conditions that cause activation (new or modified behavior) in each block. Propagation arrows specify the directionality of propagation using two different conventions:

- Unidirectional Propagation, which provides a single arrow in the direction of propagation. Conditions causing changes in behavior in the direction of the arrow may be overlaid on top of the arrow, while conditions which cause changes in the reverse direction of the arrow may be specified with an (r) at the end of the test. This enables the tractable representation of coupled interactions where there is a defined sequence (e.g., energy flows in one direction from a battery to a light bulb) but interactions may flow in both directions (e.g., a bulb burning out cuts power use).
- N-Directional Propagation, which provides arrows in both directions. This convention can be used to specify polycentric behavioral interactions (e.g., communications) where there is not single obvious direction of flow. In this convention, the direction of propagation is instead defined for each condition annotation using the convention above. When a condition in a block causes a propagation via the flow, it is writen as `[Condition]>o`, while a propagated condition in a flow that causes the activation of a block is writen as `(Condition)>[]`.


Architectures
-------------

Architectures are used to represent the structure and interaction/propagation behavior of blocks via flows and activiations. Architectures are compositions of `Primitives`_ that may be used to analyze the interactions between blocks.

Functional architectures specify the interactions between `Blocks`_ in a system.

A complete architecture diagram has:
- A full accounting of `Blocks`_ of the diagram type (Function, Action, or Component) at the desired level of abstraction, with the appropriate `Block Annotations`_ desired for the analysis;
- `Flows`_ that connect each of the functions, if any; and
- `Relationships`_ (`Propagation`_ or `Activation`_ and `Connection`_ arrows) that relate `Blocks`_ and `Flows`_ with each other.

There are three major types of architectures--Functional Architectures, Action Architectures, and Component Architectures, described next.

Functional Architectures
""""""""""""""""""""""""

Functional Architectures specify the interactions between the abstract functionalities (or, functions) to be embodied by the system. 


Diagram Types
'''''''''''''
While functional architectures can be developed using a range of conventions, two major diagram types are provided for managing scope: The Function-In-Context Diagram and the Function Architecture Diagram.

Function-in-Context Diagram
...........................

.. figure:: figures/frdl/diagrams/frdl_ficd_singleprop.svg
   :width: 800
   :alt: Function in Context Diagram

The Function-in-Context Diagram describes the system as a single function that interacts with a number of external functions representing the external socio-technical environment of the system. For example, in the above image, the function "Function Name" is controled by signal input via "External Signals" from "Control Function Name" and takes in Material and Signal flow inputs. It then produces "External Energy" (evacuated as waste) as well as an output material.


Function Architecture Diagram
.............................

.. figure:: figures/frdl/diagrams/frdl_fad_singleprop.svg
   :width: 800
   :alt: Function Architecture Diagram

The Function architecture diagram in turn describes the decomposition of the overall system function into further functions along with their interactions). For example, in the above image, the function "Function A" produces "Control Signal", which controls the energy supplied by "Function B" that in term modifies the material in "Function C", which also produces a "waste energy out" flow.


Conventions
'''''''''''
Functional Architectures can be represented using a range of conventions for representing behavioral interactions, depending on what is desired by the anlaysis. In general, the goal should be clarity, which, when there are many interactions, often means trying to represent each behavioral interaction with as little information possible. When there are few interactions, however, using a more detailed representation may be helpful for explaining an interaction in detail.


 .. |frdl_fad_separate| image:: ../docs-source/figures/frdl/diagrams/frdl_fad_separate.svg
 .. |frdl_fad_nprop| image:: ../docs-source/figures/frdl/diagrams/frdl_fad_nprop.svg
 .. |frdl_fad_singleprop| image:: ../docs-source/figures/frdl/diagrams/frdl_fad_singleprop.svg
 .. |frdl_ficd_separate| image:: ../docs-source/figures/frdl/diagrams/frdl_ficd_separate.svg
 .. |frdl_ficd_nprop| image:: ../docs-source/figures/frdl/diagrams/frdl_ficd_nprop.svg
 .. |frdl_ficd_singleprop| image:: ../docs-source/figures/frdl/diagrams/frdl_ficd_singleprop.svg

 +----------------------------------+---------------------------------------+----------------------------+----------------------------------+
 | Diagram Example                  + Seperate Connections and Activations  + N-Directional Propagations + Single-Directional Propagations  |
 +----------------------------------+---------------------------------------+----------------------------+----------------------------------+
 | Function-In Context Diagram      + |frdl_ficd_separate|                  + |frdl_ficd_nprop|          + |frdl_ficd_separate|             |
 +----------------------------------+---------------------------------------+----------------------------+----------------------------------+
 | Function Architecture Diagram    + |frdl_ficd_separate|                  +  |frdl_fad_nprop|          + |frdl_fad_singleprop|            |
 +----------------------------------+---------------------------------------+----------------------------+----------------------------------+

The table above shows three different possible conventions to apply for representing behavioral interactions using the relationships provided by FRDL, on the two example architecture diagram. As shown, the are three conventions that can be applied:
- Seperate Connections and Activations, in which the flow connections and activations are provided as seperate arrows (rather than aggregated as combined propagation arrows). Generally, this representation is to be avoided unless the relationships are not possible to be represented with a propagation arrow, or there is a desired to show specific details that would not be clear otherwise. This approach is to be avoided because it creates many more opportunites for edges to overalap with each other, making a model difficult to read the diagram clearly as more blocks and relationships are added.
- N-Directional Propagations, in which the edges are represented using N-Directional Propagation arrows. In general, this approach is to be avoided and instead N-Directional propagation arrows are only to be used for interactions that are truly multi-directional (e.g., communications or interactions with a dynamic shared environment). This is to avoid placing too many annotations on the diagram, which can make it difficult to read.
- Single-Directional Propagations, in which the edges are represented using single-directional propagation arrows. This convention is more concise than the others, while preserving the same propagation information, and even the possibility for bi-directional propagations via the "(r)" reverse propagation labels. This approach is thus preferred unless N-directional propagations present or there are specific details that need to be shown in detail for communication purposes.


Action Architectures
""""""""""""""""""""

.. figure:: figures/frdl/diagrams/frdl_actionarchitecture.svg
   :width: 800
   :alt: Action Architecture Diagram

Action architectures specify the sequence and interactions between logical actions performed by the system. They may be considered as similar to state machine diagrams as well as activity diagrams, except with the explicit specification of the means of interaction via flows. For example, in the diagram above, the system starts at "Action 1", which affects "Flow 2", then procedes to "Action 2" which modifies "Flow 2," and then either (if Action 2 fails) procedes to Action 4 followed by Action 1, or (if "Action 2" Completes) procedes to "Action 3," which uses "Flow 2."

In general, action architectures may be represented using seperate connections and activations, rather than propagation arrows. This is because actions are meant to represent discrete events in which one action leads into the next (which activate based on sequence), rather than continuously-interacting functionality (which activate via physical constraints).

Component Architectures
"""""""""""""""""""""""

.. figure:: figures/frdl/diagrams/frdl_componentarchitecture.svg
   :width: 800
   :alt: Component Architecture Diagram


Component architectures specify the interactions between components in a system. While component architectures can be modelled with a variety of conventions, in the FRDL methodology they are generally used to represent the interactions between components fulfilling a particular function (though more uses are possible). This is to help manage the complexity of interactions, since components (and component architectures) can fulfill a range of functions with complex, multidisciplinary behaviors that may be difficult to represent all on a single diagram. Instead, different component component architecture diagrams should constructed be for each function showing the component interactions involved in fulfilling that function. Then, the propagation behavior for component faults can be traced first to the immediate effects at each function level (where the component is present), and then traced at the function architecture level across the system.

Usage
-----

FRDL can be used to analyse behavioral interactions in a system, focusing mainly on the analysis of hazards. The procedure used to model and analyze hazards in a system using FRDL is as follows:

Modelling
'''''''''''

Modelling is meant to be an iterative process in which the model is developed and revised as information (e.g., behaviors, design decisions, elicited hazards) is elicited and modified. For a given architecture, the following procedure may be used:

.. figure:: figures/frdl/concepts/modelling_process.svg
   :width: 800
   :alt: FRDL Modelling Process

In this approach, the modeller first identifies functions and flows, stitches them together with relationships, and then annotates them with detailed type, behavior, and propagation information.

Analysis of the model will often lead to refinement, and this is an important part of the modelling process in a couple ways:
- First, analysis can uncover functions or interactions that were specified in ways that lead to fallacious results.
- Second, analysis of specific sencarios can help the modeller identify functions, flows, and relationships (and details of each of these), that may not have been present in their initial mental model of the system.
As such, it is expected that analysis should lead to model refinement as a key component of ensuring that it achieves and maintains an acceptable level of analytical rigor.


Developing a Model Hierarchy
............................

FRDL is meant to enable modelling activities to proceed throughout the system development process as the system becomes more detailed. The three tools for this are model refinement, annotation, and modelling hierarchy. A model hierarchy is a set of models that represent aspects of the same system at different levels of abstraction. These levels of abstraction enable the system design to include in more detail over time, while keeping the same high-level representations needed to analyse the system as a whole.

At its most detailed, a model hierarchy will include:

- A function-in-context diagram (developed first) showing how the system will interact user(s) and the environment
- A function architecture diagram (developed second) breaking the system into its high-level functions and showing the interactions between these functions
- Subsequent function architecture diagrams needed to provide more detail for any of the high-level functions
- Action architecture diagrams to specify the behavior of functions with logical behavior (e.g., users, operators, and control algorithms)
- Component architecture diagrams needed to show how functions are embodied as well as the architectural details (e.g., redundancies, behaviors, etc.) used to achieve these functions


Analysis
'''''''''''




Guide
^^^^^

Concepts
--------

Function/Action Relationship
''''''''''''''''''''''''''''

.. figure:: figures/frdl/concepts/functions_vs_actions.svg
   :width: 800
   :alt: FRDL Block Classes

Function/Component Relationship
'''''''''''''''''''''''''''''''

.. figure:: figures/frdl/concepts/functions_vs_components.svg
   :width: 800
   :alt: FRDL Block Classes


Object Orientation
''''''''''''''''''

Flows as Nodes vs edges

Functions, Actions, and Components

Correspondence with other methods
---------------------------------

Versus F/FA

Versus STPA

Versus SysML

Versus Functional Basis

Versus ARP-4761B


Examples
--------

Bread Making

Circuit (move to eps folder)

ASG 

Others
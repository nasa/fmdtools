FRDL Version 0.7
================


Overview
^^^^^^^^

FRDL stands for the Functional Reasoning Design Language, which is a graphical language for describing the functional architecture of a system. A functional architecture describes the high-level functionality (or, functions) that is to be embodied by a system, along with the interactions between these functionalities. Diagrams of functional architectures (like those enabled by FRDL) enable *functional reasoning* by providing a means to abstract away irrelevent aspects of system structure and behavior while focusing on the key (that is, functional) elements. Functional reasoning can support a range of design and analysis activities, because it helps encourage clear reasoning about how a system operates and what it is supposed to do at a high level. This can facilitate many important design and analysis activities, including product design (and especially redesign), behavioral simulation, and hazard analysis.

What can you use FRDL for?
--------------------------

As a language, FRDL can be used to support a range of activities where functional reasoning can be helpful, and is thus meant as a drop-in replacement to many existing functional modelling languages (such as block diagrams, the Functional Basis, etc.). However, the main focus of FRDL has been improving hazard analysis through its explicit representation of behavioral interactions. In the context of hazard analysis, FRDL models can be used to identify potential causes and effects of hazardous conditions, by tracing the interactions between functions that could induce or be affected by these conditions.

Why FRDL?
---------

FRDL was developed to improve the analytical rigor of the diagrams that analysts typically use to perform hazard analysis to better support causal reasoning. Compared to other languages, it has a few key features and advantages:

- FRDL encourages functional reasoning, which is a key and long-appreciated part of hazard analysis. Many graphical languages, however, do not have a good idea of what a "function" is, which can lead to confusion and ultimately poor reasoning.

- FRDL encourages (or rather, requires) the consideration of behavioral interactions. These behavioral interactions are key for understanding both (1) how hazardous conditions can lead to downsteam hazards and (2) how hazarous conditions can arise in a system. Without this explicit representation of behavioral interactions, analysts are left to either (1) use very simplistic reasoning based on what models do represent (e.g., if a function within another function fails, the containing function must also fail) or (2) rely on their own mental models of system behavior (which at the very least lack traceability and justification, and at the very most could be undetectably flawed).

- FRDL offers a scalable representation of system complexity that enables the representation of diverse interactions in highly complex systems. Unlike other methods--such as STAMP and block diagrams, which encode interactions as arrows, these interactions are represented in FRDL as nodes, making it possible to more scalably represent diverse interactions in a system that affect more than one function. This approach also encourages the grouping of like interactions, as well as the representation of wide-reaching interactions such as communications in a system-of-systems.

- FRDL explicitly supports annotations to support more detailed information to be used in analysis when it is available. Specifically, FRDL annotations can be used to better represent the dynamics of behavior (e.g., when functionality is active versus inactive) to inform the analysis of system resilience to hazards.

- FRDL enables the direct integration with the fmdtools simulation library, meaning that hazard models in FRDL can be directly implemented in simulation (rather than what is typical--simulations of system behavior looking much different from the original functional model).


Relationship with fmdtools
--------------------------

Specification
^^^^^^^^^^^^^

Primitives
----------

Architectures
-------------

Analysis Procedure
------------------


Guide
^^^^^

Concepts
--------

OOP

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
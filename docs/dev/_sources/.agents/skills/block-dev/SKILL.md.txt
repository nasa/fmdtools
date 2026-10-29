---
name: block-dev
description: Develop models of single behavior within a block
license: MIT
metadata:
  audience: users
---

## What I do and when to use me

- Create simple models with a single, main behavior and minimal interactions
- When `Function`, `Action` or `Component` classes are to be used.

## Instructions

- Unless contained in a larger `ActionArchitecture` or `ComponentArchitecture`, the default block is `Function`

- See the relevant code template in [Function Code Template](../../../docs-source/Intro_to_fmdtools.md#function-code-template)

- See the relevant examples in `/examples` for usage within a larger architecture. Importantly, when featured in a larger architecture, functions need to be connected via flows to enable behaviors to propagate between them. See the [Flow Code Template](../../../docs-source/Intro_to_fmdtools.md#flow-code-template)

- Always define containers externally, rather than defining them as inner classes of their containing flows or blocks. Then attach to Block classes using the `container_x = ClassName` syntax, where `x` is a defined letter for the given container type (s for State, m for Mode, t for Time, p for Parameter, etc.) 

- Use the relevant `Container` classes to hold local properties of the block per the relevant code templates in [Container Code Template](../../../docs-source/Intro_to_fmdtools.md#containers---the-building-blocks-of-simulations-smaller)
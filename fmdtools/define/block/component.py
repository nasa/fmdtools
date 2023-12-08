# -*- coding: utf-8 -*-
"""
Description: A module to define Components.
"""
from fmdtools.define.block.base import Block


class Component(Block):
    """Superclass for components (most attributes inherited from Block superclass)."""

    def behavior(self, time):
        """
        Placeholder for component behavior methods.

        Enables one to include components without yet having a defined behavior for
        them.
        """
        return 0

    def get_typename(self):
        return "Component"

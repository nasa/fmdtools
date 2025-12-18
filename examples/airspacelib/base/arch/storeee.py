# -*- coding: utf-8 -*-
"""
Store and supply electricity function.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The "Fault Model Design tools - fmdtools version 2" software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE/2.0. 

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from fmdtools.define.block.function import Function
from fmdtools.define.container.mode import Mode


from examples.airspacelib.base.arch.flows import Force, Electricity


class StoreEEMode(Mode):

    opermodes = ('charged', 'in_use', 'off')
    mode: str = 'charged'
    fault_break = ()
    fault_depletion: dict = {'disturbances': (('electricity.s.charge', 25.0), ),
                             'phases': ('in_use', 1.0)}


class StoreAndSupplyElectricity(Function):
    """Function used to store and supply energy to the drone."""

    __slots__ = ('force', 'electricity')
    flow_force = Force
    flow_electricity = Electricity
    container_m = StoreEEMode

    def static_behavior(self):
        """
        Execute power supply static behavior.

        Determine faults as well as electricity voltage/current.
        """
        if self.force.s.contact_support >= 5.0:
            self.m.add_fault('break')

        if self.electricity.s.charge <= 0.0 or self.m.has_fault('break'):
            self.electricity.s.charge = 0.0
            self.electricity.s.put(voltage_high=False, voltage_low=False)
        elif self.electricity.s.charge > 5.0:
            self.electricity.s.voltage_low = float(self.electricity.s.power_low)
            self.electricity.s.voltage_high = float(self.electricity.s.power_high)
        elif self.electricity.s.charge > 0.0:
            self.electricity.s.voltage_low = float(self.electricity.s.power_low)
            self.electricity.s.voltage_high = False

    def dynamic_behavior(self):
        """
        Execute power supply dynamic behavior.

        Determines energy use and thus charge.
        """
        rate_high = self.electricity.s.mul('current_high', 'voltage_high')
        rate_low = self.electricity.s.mul('current_low', 'voltage_low')
        if rate_high > 0.0:
            self.m.set_mode('in_use')
        else:
            self.m.set_mode('off')
        self.electricity.s.inc(charge=-(rate_high+0.1*rate_low))


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

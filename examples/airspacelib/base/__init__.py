# -*- coding: utf-8 -*-
"""
Base packages for the Airspace Library.

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

import examples.airspacelib.base.aircraft as aircraft
import examples.airspacelib.base.parameter as parameter
import examples.airspacelib.base.state as state
import examples.airspacelib.base.arch as arch

__all__ = ['aircraft', 'parameter', 'state', 'arch']
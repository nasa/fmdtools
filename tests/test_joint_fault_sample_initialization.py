#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Regression tests for complete joint fault sample initialization.

Copyright © 2024, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The “"Fault Model Design tools - fmdtools version 2"” software is licensed
under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import pytest

from fmdtools.analyze.phases import PhaseMap, join_phasemaps
from fmdtools.define.architecture.function import ExFxnArch
from fmdtools.sim import propagate
from fmdtools.sim.sample import FaultDomain, FaultSample, JointFaultSample


@pytest.fixture
def domains():
    model = ExFxnArch(sp={"end_time": 6.0})
    first, second = FaultDomain(model), FaultDomain(model)
    first.add_fault("ex_fxn", "low")
    second.add_fault("ex_fxn2", "no_charge")
    return first, second


def phase_options(mode):
    if mode == "disabled":
        return {"def_mdl_phasemap": False}
    if mode == "joined":
        return {
            "phasemaps": [PhaseMap({"first": [0, 4]}), PhaseMap({"second": [2, 6]})]
        }
    if mode == "single":
        return {"phasemaps": [PhaseMap({"first": [0, 4]})]}
    return {}


@pytest.mark.parametrize("mode", ["default", "disabled", "joined", "single"])
@pytest.mark.parametrize("domain_count", [1, 2])
def test_joint_sample_starts_empty_with_merged_domains(domains, mode, domain_count):
    selected = domains[:domain_count]
    options = phase_options(mode)
    sample = JointFaultSample(*selected, **options)
    assert sample.scenarios() == []
    assert sample.get_times() == []
    assert sample.num_scenarios() == 0
    assert sample.named_scenarios() == {}
    assert sample.faultdomain.mdl is selected[0].mdl
    expected_faults = {
        key: fault for domain in selected for key, fault in domain.faults.items()
    }
    assert sample.faultdomain.faults == expected_faults
    if mode == "disabled":
        assert not sample.phasemap
    elif mode in ("joined", "single"):
        expected = join_phasemaps(*options["phasemaps"])
        assert sample.phasemap.phases == expected.phases
    else:
        assert sample.phasemap.phases == PhaseMap(selected[0].mdl.sp.phases).phases


@pytest.mark.parametrize("mode", ["default", "disabled", "joined"])
@pytest.mark.parametrize("n_joint", [1, 2])
def test_sampling_matches_an_equivalent_regular_fault_sample(domains, mode, n_joint):
    options = phase_options(mode)
    combined = FaultDomain(domains[0].mdl)
    for domain in domains:
        combined.faults.update(domain.faults)
    phasemap = join_phasemaps(*options["phasemaps"]) if "phasemaps" in options else {}
    reference = FaultSample(
        combined, phasemap=phasemap, def_mdl_phasemap=mode != "disabled"
    )
    joint = JointFaultSample(*domains, **options)
    reference.add_fault_times([2, 3], n_joint=n_joint)
    joint.add_fault_times([2, 3], n_joint=n_joint)
    assert sorted(joint.get_times()) == sorted(reference.get_times()) == [2, 3]
    assert [scen.asdict() for scen in joint.scenarios()] == [
        scen.asdict() for scen in reference.scenarios()
    ]
    assert joint.named_scenarios().keys() == reference.named_scenarios().keys()


def test_separate_instances_do_not_share_sampling_state(domains):
    first = JointFaultSample(*domains)
    second = JointFaultSample(*domains)
    original_domains = [dict(domain.faults) for domain in domains]
    first.add_single_fault_scenario(next(iter(domains[0].faults)), 1)
    first.faultdomain.faults.clear()
    assert second.scenarios() == []
    assert second.get_times() == []
    assert second.faultdomain.faults
    assert [domain.faults for domain in domains] == original_domains


def test_empty_domains_have_usable_empty_state():
    model = ExFxnArch()
    sample = JointFaultSample(FaultDomain(model), def_mdl_phasemap=False)
    sample.add_fault_times([1, 2])
    assert sample.scenarios() == []
    assert sample.get_times() == []
    assert not sample.phasemap


def test_joint_scenarios_can_be_propagated(domains):
    sample = JointFaultSample(*domains)
    sample.add_fault_times([2], n_joint=2)
    results, histories = propagate.fault_sample(
        domains[0].mdl, sample, showprogress=False
    )
    name = sample.scenarios()[0].name
    assert name in results.nest(levels=1)
    assert name in histories.nest(levels=1)
    assert "nominal" in histories.nest(levels=1)


@pytest.mark.parametrize("use_model_phases", [False, True])
def test_regular_fault_sample_initialization_is_unchanged(domains, use_model_phases):
    sample = FaultSample(domains[0], def_mdl_phasemap=use_model_phases)
    assert sample.scenarios() == []
    assert sample.get_times() == []
    assert bool(sample.phasemap) == use_model_phases

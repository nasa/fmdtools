"""Regression tests for distinct maps with matching phase labels."""

import copy
import itertools

import pytest

from fmdtools.analyze.phases import PhaseMap, join_phasemaps


CASES = [
    ({"run": [0, 1]}, {"run": [5, 6]}),
    ({"run": [0, 3]}, {"run": [2, 5]}),
    ({"run": [2, 3]}, {"run": [0, 5]}),
    ({"run": [0, 2]}, {"run": [2, 5]}),
    ({"run": [0, 4]}, {"other": [1, 3]}, {"run": [2, 6]}),
    ({"run": [0, 1], "stop": [2, 5]}, {"run": [3, 4], "stop": [5, 6]}),
    ({"first": [0, 3]}, {"second": [2, 5]}),
    ({"run": [1, 3]}, {"run": [1, 3]}),
]


def _expected(maps):
    result = {}
    for labels in itertools.product(*(mapping.phases for mapping in maps)):
        # Enumerating each map's integer sample times supplies an independent
        # intersection oracle, without using the production interval helper.
        times = [
            set(range(mapping.phases[label][0], mapping.phases[label][1] + 1))
            for mapping, label in zip(maps, labels)
        ]
        intersection = set.intersection(*times)
        if intersection:
            result[labels] = [min(intersection), max(intersection)]
    return result


@pytest.mark.parametrize("definitions", CASES)
def test_joint_phases_include_every_source_interval(definitions):
    maps = [PhaseMap(copy.deepcopy(definition)) for definition in definitions]
    original = [copy.deepcopy(mapping.phases) for mapping in maps]
    combined = join_phasemaps(*maps)
    expected = _expected(maps)
    assert combined.phases == expected
    assert combined.modephases == {}
    assert [mapping.phases for mapping in maps] == original
    for labels, (start, end) in expected.items():
        assert combined.get_phase_times(labels) == list(range(start, end + 1))
        assert combined.calc_phase_time(labels) == end - start + 1


@pytest.mark.parametrize("definitions", CASES[:6])
def test_permuting_maps_only_permutates_joint_labels(definitions):
    maps = [PhaseMap(definition) for definition in definitions]
    expected = _expected(maps)
    for order in itertools.permutations(range(len(maps))):
        combined = join_phasemaps(*(maps[index] for index in order))
        reordered = {
            tuple(labels[index] for index in order): interval
            for labels, interval in expected.items()
        }
        assert combined.phases == reordered


def test_two_disjoint_same_named_phases_produce_no_sample_times():
    combined = join_phasemaps(PhaseMap({"run": [0, 1]}), PhaseMap({"run": [5, 6]}))
    assert combined.phases == {}
    assert combined.get_sample_times() == {}


def test_single_map_keeps_tuple_labels_and_intervals():
    assert join_phasemaps(PhaseMap({"run": [0, 1]})).phases == {("run",): [0, 1]}


def test_empty_source_map_produces_no_joint_phases():
    assert join_phasemaps(PhaseMap({}), PhaseMap({"run": [0, 1]})).phases == {}

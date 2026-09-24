"""Regression tests for joining phase maps with repeated labels."""

import copy
import itertools
import unittest

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


class JoinPhaseMapsRepeatedLabelsTests(unittest.TestCase):
    """Verify repeated labels do not overwrite source phase intervals."""

    def test_repeated_labels_preserve_all_source_intervals(self):
        for definitions in CASES:
            with self.subTest(definitions=definitions):
                maps = [
                    PhaseMap(copy.deepcopy(definition)) for definition in definitions
                ]
                original = [copy.deepcopy(mapping.phases) for mapping in maps]
                combined = join_phasemaps(*maps)
                expected = _expected(maps)

                self.assertEqual(combined.phases, expected)
                self.assertEqual(combined.modephases, {})
                self.assertEqual([mapping.phases for mapping in maps], original)
                for labels, (start, end) in expected.items():
                    self.assertEqual(
                        combined.get_phase_times(labels),
                        list(range(start, end + 1)),
                    )
                    self.assertEqual(
                        combined.calc_phase_time(labels),
                        end - start + 1,
                    )

    def test_map_order_only_reorders_joint_labels(self):
        for definitions in CASES[:6]:
            maps = [PhaseMap(definition) for definition in definitions]
            expected = _expected(maps)
            for order in itertools.permutations(range(len(maps))):
                with self.subTest(definitions=definitions, order=order):
                    combined = join_phasemaps(*(maps[index] for index in order))
                    reordered = {
                        tuple(labels[index] for index in order): interval
                        for labels, interval in expected.items()
                    }
                    self.assertEqual(combined.phases, reordered)

    def test_disjoint_repeated_labels_have_no_samples(self):
        combined = join_phasemaps(
            PhaseMap({"run": [0, 1]}),
            PhaseMap({"run": [5, 6]}),
        )
        self.assertEqual(combined.phases, {})
        self.assertEqual(combined.get_sample_times(), {})

    def test_single_map_preserves_tuple_label_and_interval(self):
        combined = join_phasemaps(PhaseMap({"run": [0, 1]}))
        self.assertEqual(combined.phases, {("run",): [0, 1]})

    def test_empty_source_map_has_no_joint_phases(self):
        combined = join_phasemaps(PhaseMap({}), PhaseMap({"run": [0, 1]}))
        self.assertEqual(combined.phases, {})


if __name__ == "__main__":
    unittest.main()

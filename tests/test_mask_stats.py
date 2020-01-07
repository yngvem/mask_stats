from mask_stats import MaskEvaluator
import numpy as np
import pytest


TEMPLATE1 = np.array(
    [
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
    ]
)


TEMPLATE2 = np.array(
    [
        [0, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
    ]
)


@pytest.fixture
def evaluator():
    me = MaskEvaluator(detection_threshold=0.5)
    me.evaluate(TEMPLATE1, TEMPLATE2)
    return me


def test_correct_num_labels(evaluator):
    assert evaluator.num_detected_objects[0] == 3
    assert evaluator.num_detected_objects[1] == 2


def test_detection_mask_has_correct_length(evaluator):
    assert sum(evaluator.detected_mask[0]) == evaluator.num_detected_objects[0]
    assert sum(evaluator.detected_mask[1]) == evaluator.num_detected_objects[1]


def test_detected_metric_has_correct_length(evaluator):
    assert len(evaluator.detected_structure_hd[0]) == evaluator.num_detected_objects[0]
    assert len(evaluator.detected_structure_hd[1]) == evaluator.num_detected_objects[1]


def test_undetected_metric_has_correct_length(evaluator):
    assert (
        len(evaluator.undetected_structure_hd[0])
        == evaluator.num_labels[0] - evaluator.num_detected_objects[0]
    )
    assert (
        len(evaluator.undetected_structure_hd[1])
        == evaluator.num_labels[1] - evaluator.num_detected_objects[1]
    )


def test_detected_dice_is_correct(evaluator):
    assert set(evaluator.detected_structure_dice[0].values()) == {1, 1, (2 / (1 + 3))}
    assert set(evaluator.detected_structure_dice[1].values()) == {1, 1}


def test_coverage_fraction_is_correct(evaluator):
    assert sorted(evaluator.coverage_fraction[0]) == [0, 1, 1, 1]
    assert sorted(evaluator.coverage_fraction[1]) == [0, 1 / 3, 1, 1]

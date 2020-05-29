from collections import Counter

import numpy as np
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import morphological_gradient, label

from mask_stats import compute_all_evaluations_for_mask_pairs

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


def test_metrics():
    compute_all_evaluations_for_mask_pairs([TEMPLATE1], [TEMPLATE2])


def test_correct_num_labels():
    eval_mask1, eval_mask2 = compute_all_evaluations_for_mask_pairs([TEMPLATE1], [TEMPLATE2])
    assert len(eval_mask1['object_wise']['mask_num']) == 4
    assert len(eval_mask2['object_wise']['mask_num']) == 4


def test_coverage_fractions():
    eval_mask1, eval_mask2 = compute_all_evaluations_for_mask_pairs([TEMPLATE1], [TEMPLATE2])
    cfracs1 = Counter(eval_mask1['object_wise']['coverage_fraction'])
    assert cfracs1[1] == 3
    assert cfracs1[0] == 1
    
    cfracs2 = Counter(eval_mask2['object_wise']['coverage_fraction'])
    assert cfracs2[1] == 2
    assert cfracs2[1/3] == 1
    assert cfracs2[0] == 1


def test_object_accuracy():
    eval_mask1, eval_mask2 = compute_all_evaluations_for_mask_pairs([TEMPLATE1], [TEMPLATE2], overlap_threshold=0.5)
    assert eval_mask1['overall']['object_accuracy'][0] == 3/4
    assert eval_mask2['overall']['object_accuracy'][0] == 2/4

    eval_mask1, eval_mask2 = compute_all_evaluations_for_mask_pairs([TEMPLATE1], [TEMPLATE2], overlap_threshold=0.1)
    assert eval_mask1['overall']['object_accuracy'][0] == 3/4
    assert eval_mask2['overall']['object_accuracy'][0] == 3/4


def test_hausdorff_distance():
    eval_mask1, eval_mask2 = compute_all_evaluations_for_mask_pairs([TEMPLATE1], [TEMPLATE2], overlap_threshold=0.5)

    structuring_el_size = tuple(3 for _ in TEMPLATE1.shape)
    grad1 = morphological_gradient(TEMPLATE1, size=structuring_el_size)
    grad2 = morphological_gradient(TEMPLATE2, size=structuring_el_size)
    grad1_nnz = np.array(np.nonzero(grad1)).T
    grad2_nnz = np.array(np.nonzero(grad2)).T

    hd_scipy1 = directed_hausdorff(grad1_nnz, grad2_nnz)[0]
    assert hd_scipy1 == eval_mask1['overall']['hausdorff_distance'][0]
    hd_scipy2 = directed_hausdorff(grad2_nnz, grad1_nnz)[0]
    assert hd_scipy2 == eval_mask2['overall']['hausdorff_distance'][0]


def test_labelled_hausdorff_distance():
    eval_mask1, eval_mask2 = compute_all_evaluations_for_mask_pairs(
        [TEMPLATE1], [TEMPLATE2], overlap_threshold=0.5
    )

    structuring_el_size = tuple(3 for _ in TEMPLATE1.shape)
    labelled_1, num_labels_1 = label(TEMPLATE1)
    object_hausdorffs_1 = []
    for label_id in range(1, num_labels_1+1):
        mask = (labelled_1 == label_id).astype(int)

        grad1 = morphological_gradient(mask, size=structuring_el_size)
        grad2 = morphological_gradient(TEMPLATE2, size=structuring_el_size)
        grad1_nnz = np.array(np.nonzero(grad1)).T
        grad2_nnz = np.array(np.nonzero(grad2)).T

        hd_scipy1 = directed_hausdorff(grad1_nnz, grad2_nnz)[0]

        object_hausdorffs_1.append(hd_scipy1)

    assert Counter(eval_mask1['object_wise']['hausdorff_distance']) == Counter(object_hausdorffs_1)

    labelled_2, num_labels_2 = label(TEMPLATE2)
    object_hausdorffs_2 = []
    for label_id in range(1, num_labels_2+1):
        mask = (labelled_2 == label_id).astype(int)

        grad1 = morphological_gradient(mask, size=structuring_el_size)
        grad2 = morphological_gradient(TEMPLATE1, size=structuring_el_size)
        grad1_nnz = np.array(np.nonzero(grad1)).T
        grad2_nnz = np.array(np.nonzero(grad2)).T

        hd_scipy2 = directed_hausdorff(grad1_nnz, grad2_nnz)[0]

        object_hausdorffs_2.append(hd_scipy2)

    assert Counter(eval_mask2['object_wise']['hausdorff_distance']) == Counter(object_hausdorffs_2)
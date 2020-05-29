from itertools import product

import numpy as np


def compute_label_overlap(labelled_1, labelled_2, label_1, label_2):
    """Compute overlap between `label_1` and `label_2`.

    Arguments
    ---------
    labelled_1 : np.ndarray(dtype=int)
        Region labelled mask
    labelled_2 : np.ndarray(dtype=int)
        Region labelled mask
    label_1 : int
        Label of relevant region in labelled_1
    label_2 : int
        Label of relevant region in labelled_2

    Returns
    -------
    overlap : int
        The number of overlapping pixels between the two masks of interest.
    """
    labelled_1 = labelled_1 == label_1
    labelled_2 = labelled_2 == label_2
    return np.sum(labelled_1 * labelled_2)


def compute_pairwise_overlaps(
    labelled_1, labelled_2, num_labels_1, num_labels_2, locs_1, locs_2
):
    """Compute the overlap between two region labelled masks.

    Compute all pairwise overlaps between two region-labelled images.

    Arguments
    ---------
    labelled_1 : np.ndarray(dtype=int)
        Region labelled mask
    labelled_2 : np.ndarray(dtype=int)
        Region labelled mask
    num_labels_1 : int
        Number of different labelled regions in labelled_1.
    num_labels_2 : int
        Number of different labelled regions in labelled_2.
    locs_1 : list[slice]
        Slices that specify the locations of the labelled regions in labelled_1
    locs_2 : list[slice]
        Slices that specify the locations of the labelled regions in labelled_2

    Returns
    -------
    overlaps : np.ndarray(dtype=int, shape=(num_labels_1, num_labels_2))
        Matrix with degree of overlap between the different labels.
    """
    overlaps = np.zeros((num_labels_1, num_labels_2))

    # Generate iterators
    region_info_1 = enumerate(locs_1)
    region_info_2 = enumerate(locs_2)

    for info_1, info_2 in product(region_info_1, region_info_2):
        # Extract info from iterators
        idx_1, location_1 = info_1
        idx_2 = info_2[0]

        # Compute overlap:
        ## We have location_1 for both of these regions since
        ## we want to discover the overlap and then it suffices
        ## to look at the bounding box for one of the objects
        ## since there will be no overlap outside this bounding box
        overlaps[idx_1, idx_2] = compute_label_overlap(
            labelled_1[location_1], labelled_2[location_1], idx_1 + 1, idx_2 + 1
        )
    return overlaps


def compute_volumes(labelled, num_labels):
    """Compute the volumes of all the region-labelled masks.

    Arguments
    ---------
    labelled : np.ndarray(dtype=int)
        Region labelled masks
    num_labels : int
        Number of unique labels in the region labelled mask

    Returns
    -------
    volumes : np.ndarray(dtype=int)
        List that contains the volume of each labelled mask. Element `i` contains
        the volume of the mask with label `i+1`.
    """
    return np.array([np.sum(labelled == label) for label in range(1, num_labels + 1)])


def compute_relative_overlaps(volumes_1, volumes_2, overlap):
    """Compute the relative overlap between each region labelled mask.

    Arguments
    ---------
    volumes_1 : np.ndarray(dtype=int, shape=(num_labels_1,))
        Size of each mask in the first region-labelled image
    volumes_2 : np.ndarray(dtype=int, shape=(num_labels_2,))
        Size of each mask in the second region-labelled image
    overlap : np.ndarray(dtype=int, shape=(num_labels_1, num_labels_2))
        Matrix with overlap between each labels of the two masks.
        Returned by `compute_pairwise_overlaps`.

    Returns
    -------
    relative_overlap_1 : np.ndarray(dtype=float, shape=(num_labels_1, num_labels_2))
    relative_overlap_2 : np.ndarray(dtype=float, shape=(num_labels_1, num_labels_2))
    """
    return (overlap / volumes_1.reshape(-1, 1), overlap / volumes_2.reshape(1, -1))


def compute_coverage_fraction(relative_overlap_1, relative_overlap_2):
    """Compute the coverage fractions.

    Arguments
    ---------
    relative_overlap_1 : np.ndarray(dtype=float, shape=(num_labels_1, num_labels_2))
        The relative overlap between each label in the first mask and each label in
        the second mask.
    relative_overlap_2 : np.ndarray(dtype=float, shape=(num_labels_1, num_labels_2))
        The relative overlap between each label in the second mask and each label in
        the fist mask.
    Returns
    -------
    coverage_fractions_1 : np.ndarray(dtype=float, shape=(num_labels_1))
        The coverage fraction of each mask in the first region labelled mask
    coverage_fractions_2 : np.ndarray(dtype=float, shape=(num_labels_2))
        The coverage fraction of each mask in the second region labelled mask
    """
    return relative_overlap_1.sum(1), relative_overlap_2.sum(0)


def compute_overall_dice(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    tp = np.sum(mask1 * mask2)

    return 2 * tp / (mask1.sum() + mask2.sum())


def compute_object_dice(overlap, volumes_1, volumes_2, threshold=0):
    """Compute the component-wise dice.

    For each connected component, C, in the first mask, we find the connected
    components in the second mask that covers a factor of at least `threshold`
    times the volume of C. Once we have these components, we compute the dice.
    """

    object_dice_1 = []
    for i, volume in enumerate(volumes_1):
        relevant_volumes_2 = 2 * overlap[i] / volume > threshold
        object_dice_1.append(
            2 * overlap[i].sum() / (volume + volumes_2[relevant_volumes_2].sum())
        )

    object_dice_2 = []
    for i, volume in enumerate(volumes_2):
        relevant_volumes_1 = overlap[:, i] / volume > threshold
        object_dice_2.append(
            2 * overlap[:, i].sum() / (volume + volumes_1[relevant_volumes_1].sum())
        )

    return object_dice_1, object_dice_2


def compute_num_detected_objects(relative_overlap_1, relative_overlap_2, threshold):
    """Compute the number of objects that are covered by a fraction over the given threshold.
    """
    detected_object_1 = np.sum(relative_overlap_1.sum(1) > threshold)
    detected_object_2 = np.sum(relative_overlap_2.sum(0) > threshold)
    return detected_object_1, detected_object_2


def compute_object_accuracy(relative_overlap_1, relative_overlap_2, threshold):
    accuracy_1 = np.mean(relative_overlap_1.sum(1) > threshold)
    accuracy_2 = np.mean(relative_overlap_2.sum(0) > threshold)
    return accuracy_1, accuracy_2

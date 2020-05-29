import numpy as np
from scipy.ndimage import morphological_gradient


def _norm_along_last_axis(x):
    """Compute the norm of x along the last axis.
    """
    return np.sqrt(np.sum(np.square(x), axis=x.ndim - 1))


def _compute_set_distances(nonzeros_1, nonzeros_2):
    """Compute all surface distances from one set to the other.
    """
    distances = np.zeros(len(nonzeros_1))
    for i, _ in enumerate(distances):
        distances[i] = np.min(
            _norm_along_last_axis(nonzeros_1[i].reshape(1, -1) - nonzeros_2)
        )
    return distances


def compute_surface_distances(mask1, mask2):
    """Return the surface distances for all points on the surface of mask1 to the surface of mask2.

    Arguments
    ---------
    mask1 : np.ndarray
        Boolean mask to compute distances from 
    mask2 : np.ndarray
        Boolean mask to compute distances to
    """
    structuring_el_size = tuple(3 for _ in mask1.shape)
    grad1 = morphological_gradient(mask1.astype(int), size=structuring_el_size)
    grad2 = morphological_gradient(mask2.astype(int), size=structuring_el_size)

    nonzeros_1 = np.array(np.nonzero(grad1)).T
    nonzeros_2 = np.array(np.nonzero(grad2)).T
    return np.sort(_compute_set_distances(nonzeros_1, nonzeros_2))


def compute_labelled_surface_distances(
    labelled_1, labelled_2, num_labels_1, num_labels_2
):
    """Compute the surface distances for for all connected components in one mask to the whole second mask.
    """
    mask1 = labelled_1 != 0
    mask2 = labelled_2 != 0

    surface_distance_label_1 = []
    for idx in range(num_labels_1):
        surface_distance_label_1.append(
            compute_surface_distances(labelled_1 == idx + 1, mask2)
        )

    surface_distance_label_2 = []
    for idx in range(num_labels_2):
        surface_distance_label_2.append(
            compute_surface_distances(labelled_2 == idx + 1, mask1)
        )

    return surface_distance_label_1, surface_distance_label_2


def compute_object_percentile_surface_distances(
    labelled_surface_distances_1, labelled_surface_distances_2, percentile
):
    """Compute the Hausdorff distance for for all connected components in one mask to the whole second mask.
    """
    hausdorffs_label_1 = []
    for surface_distance in labelled_surface_distances_1:
        hausdorffs_label_1.append(np.percentile(surface_distance, percentile))

    hausdorffs_label_2 = []
    for surface_distance in labelled_surface_distances_2:
        hausdorffs_label_2.append(np.percentile(surface_distance, percentile))
    return np.array(hausdorffs_label_1), np.array(hausdorffs_label_2)


def compute_overall_percentile_surface_distances(
    labelled_surface_distances_1, labelled_surface_distances_2, percentile
):
    hausdorff_1 = np.percentile(
        np.concatenate(labelled_surface_distances_1), percentile
    )
    hausdorff_2 = np.percentile(
        np.concatenate(labelled_surface_distances_2), percentile
    )

    return hausdorff_1, hausdorff_2


def compute_object_average_surface_distances(labelled_surface_distances_1, labelled_surface_distances_2):
    """Compute the Hausdorff distance for for all connected components in one mask to the whole second mask.
    """
    asd_label_1 = []
    for surface_distance in labelled_surface_distances_1:
        asd_label_1.append(np.mean(surface_distance))

    asd_label_2 = []
    for surface_distance in labelled_surface_distances_2:
        asd_label_2.append(np.mean(surface_distance))

    return (
        np.array(asd_label_1),
        np.array(asd_label_2),
    )


def compute_overall_average_surface_distances(labelled_surface_distances_1, labelled_surface_distances_2):
    asd_1 = np.mean(np.concatenate(labelled_surface_distances_1))
    asd_2 = np.mean(np.concatenate(labelled_surface_distances_2))

    return asd_1, asd_2

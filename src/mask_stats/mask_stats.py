from collections import defaultdict
from itertools import product
from numbers import Number
from typing import Dict, List, NamedTuple

import numpy as np
from scipy.ndimage import find_objects, label, morphological_gradient
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm


__all__ = ["EvaluationOutput", "MaskEvaluator", "MasksEvaluator"]


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
        idx_2, location_2 = info_2

        # Compute overlap
        overlaps[idx_1, idx_2] = compute_label_overlap(
            labelled_1[location_1], labelled_2[location_1], idx_1 + 1, idx_2 + 1
        )
    return overlaps


def compute_areas(labelled, num_labels):
    """Compute the areas of all the region-labelled masks.

    Arguments
    ---------
    labelled : np.ndarray(dtype=int)
        Region labelled masks
    num_labels : int
        Number of unique labels in the region labelled mask

    Returns
    -------
    areas : np.ndarray(dtype=int)
        List that contains the area of each labelled mask. Element `i` contains
        the area of the mask with label `i+1`.
    """
    return np.array([np.sum(labelled == label) for label in range(1, num_labels + 1)])


def compute_relative_overlap(areas_1, areas_2, overlap):
    """Compute the relative overlap between each region labelled mask.

    Arguments
    ---------
    areas_1 : np.ndarray(dtype=int, shape=(num_labels_1,))
        Size of each mask in the first region-labelled image
    areas_2 : np.ndarray(dtype=int, shape=(num_labels_2,))
        Size of each mask in the second region-labelled image
    overlap : np.ndarray(dtype=int, shape=(num_labels_1, num_labels_2))
        Matrix with overlap between each labels of the two masks.
        Returned by `compute_pairwise_overlaps`.

    Returns
    -------
    relative_overlap_1 : np.ndarray(dtype=float, shape=(num_labels_1, num_labels_2))
    relative_overlap_2 : np.ndarray(dtype=float, shape=(num_labels_1, num_labels_2))
    """
    return (overlap / areas_1.reshape(-1, 1), overlap / areas_2.reshape(1, -1))


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


def _norm(x):
    """Compute the norm of x along the last axis.
    """
    return np.sqrt(np.sum(np.square(x), axis=x.ndim - 1))


def _set_distances(nonzeros_1, nonzeros_2):
    """Compute all surface distances from one set to the other.
    """
    distances = np.zeros(len(nonzeros_1))
    for i, _ in enumerate(distances):
        distances[i] = np.min(_norm(nonzeros_1[i].reshape(1, -1) - nonzeros_2))
    return distances


def percentile_directed_hd(nonzeros_1, nonzeros_2, percentile):
    distances = _directed_distances(nonzeros_1, nonzeros_2)
    return np.percentile(distances, percentile)


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
    return np.sort(_set_distances(nonzeros_1, nonzeros_2))


def compute_labelled_surface_distances(labelled_1, labelled_2, num_labels_1, num_labels_2):
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


def structure_hd(
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


def overall_hd(
    labelled_surface_distances_1, labelled_surface_distances_2, percentile
):
    hausdorff_1 = np.percentile(np.concatenate(labelled_surface_distances_1), percentile)
    hausdorff_2 = np.percentile(np.concatenate(labelled_surface_distances_2), percentile)

    return hausdorff_1, hausdorff_2


def structure_asd(
    labelled_surface_distances_1, labelled_surface_distances_2
):
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


def overall_asd(
    labelled_surface_distances_1, labelled_surface_distances_2
):
    asd_1 = np.mean(np.concatenate(labelled_surface_distances_1))
    asd_2 = np.mean(np.concatenate(labelled_surface_distances_2))

    return asd_1, asd_2


def overall_dice(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    tp = np.sum(mask1*mask2)

    return 2*tp/(mask1.sum() + mask2.sum())

def structure_dice(overlap, areas_1, areas_2, threshold=0):
    """Compute the component-wise dice.

    For each connected component, C, in the first mask, we find the connected
    components in the second mask that covers a factor of at least `threshold`
    times the area of C. Once we have these components, we compute the dice.
    """

    structure_dice_1 = []
    for i, area in enumerate(areas_1):
        relevant_areas_2 = 2 * overlap[i] / area > threshold
        structure_dice_1.append(
            2 * overlap[i].sum() / (area + areas_2[relevant_areas_2].sum())
        )

    structure_dice_2 = []
    for i, area in enumerate(areas_2):
        relevant_areas_1 = overlap[:, i] / area > threshold
        structure_dice_2.append(
            2 * overlap[:, i].sum() / (area + areas_1[relevant_areas_1].sum())
        )

    return structure_dice_1, structure_dice_2


def num_detected_objects(relative_overlap_1, relative_overlap_2, threshold):
    """Compute the number of objects that are covered by a fraction over the given threshold.
    """
    detected_object_1 = np.sum(relative_overlap_1.sum(1) > threshold)
    detected_object_2 = np.sum(relative_overlap_2.sum(0) > threshold)
    return detected_object_1, detected_object_2


class EvaluationOutput(NamedTuple):
    summary_statistics: Dict[str, List[Number]]
    detected_metrics: Dict[str, List[Number]]
    undetected_metrics: Dict[str, List[Number]]


class MaskEvaluator:
    """Compute performance metrics between two binary masks.
    """

    def __init__(self, dice_threshold=0, detection_threshold=0.3, names=None):
        """

        Arguments
        ---------
        dice_threshold : float
            Fraction of the the blob that need to be covered by each blob used
            when computing the dice.
        detection_threshold : float
            Fraction of a blob needed to be covered for it to be detected.
        names : Indexable [length=2]
            Names of the two masks.
        """
        if names is None:
            names = [0, 1]
        self.names = names

        self._dice_threshold = dice_threshold
        self._detection_threshold = detection_threshold
        self._cache = {}

    @property
    def detection_threshold(self):
        return self._detection_threshold

    @detection_threshold.setter
    def detection_threshold(self, value):
        self._detection_threshold = value

    @property
    def dice_threshold(self):
        return self._dice_threshold

    @dice_threshold.setter
    def dice_threshold(self, value):
        self._should_recompute["dice"] = True
        self._dice_threshold = value

    def _reset_cache(self):
        self._cache = {}

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, value):
        self._cache = {}
        self._masks = tuple(value)

    @property
    def labelled(self):
        if 'labelled' not in self._cache:
            labelled0, num_labels0 = label(self.masks[0])
            labelled1, num_labels1 = label(self.masks[1])
            self._cache['labelled'] = (labelled0, labelled1)
            self._cache['num_labels'] = (num_labels0, num_labels1)
        return self._cache['labelled']

    @property
    def num_labels(self):
        if 'num_labels' not in self._cache:
            labelled0, num_labels0 = label(self.masks[0])
            labelled1, num_labels1 = label(self.masks[1])
            self._cache['labelled'] = (labelled0, labelled1)
            self._cache['num_labels'] = (num_labels0, num_labels1)
        return self._cache['num_labels']

    @property
    def locations(self):
        if 'location' not in self._cache:
            self._cache['locations'] = tuple(
                find_objects(labelled) for labelled in self.labelled
            )
        return self._cache['locations']

    @property
    def overlap(self):
        if 'overlap' not in self._cache:
            self._cache['overlap'] = compute_pairwise_overlaps(
                *self.labelled, *self.num_labels, *self.locations
            )
        return self._cache['overlap']

    @property
    def areas(self):
        if 'areas' not in self._cache:
            self._cache['areas'] = (
                compute_areas(self.labelled[0], self.num_labels[0]),
                compute_areas(self.labelled[1], self.num_labels[1]),
            )
        return self._cache['areas']

    @property
    def relative_overlap(self):
        if 'relative_overlap' not in self._cache:
            self._cache['relative_overlap'] = compute_relative_overlap(*self.areas, self.overlap)
        return self._cache['relative_overlap']

    @property
    def surface_distances(self):
        if 'surface_distances' not in self._cache:
            self._cache['surface_distances'] = compute_labelled_surface_distances(
                *self.labelled, *self.num_labels
            )
        return self._cache['surface_distances']

    @property
    def structure_hd(self):
        if 'structure_hd' not in self._cache:
            self._cache['structure_hd'] = structure_hd(
                *self.surface_distances, percentile=100
            )
        return self._cache['structure_hd']

    @property
    def structure_hd95(self):
        if 'structure_hd95' not in self._cache:
            self._cache['structure_hd95'] = structure_hd(
                *self.surface_distances, percentile=95
            )
        return self._cache['structure_hd95']

    @property
    def structure_asd(self):
        if 'structure_asd' not in self._cache:
            self._cache['structure_asd'] = structure_asd(
                *self.surface_distances
            )
        return self._cache['structure_asd']

    @property
    def structure_msd(self):
        if 'structure_msd' not in self._cache:
            self._cache['structure_msd'] = structure_hd(
                *self.surface_distances, percentile=50
            )
        return self._cache['structure_msd']

    @property
    def structure_msd(self):
        if 'structure_msd' not in self._cache:
            self._cache['structure_msd'] = structure_hd(
                *self.surface_distances, percentile=50
            )
        return self._cache['structure_msd']

    @property
    def overall_hd(self):
        if 'overall_hd' not in self._cache:
            self._cache['overall_hd'] = overall_hd(
                *self.surface_distances, percentile=100
            )
        return self._cache['overall_hd']

    @property
    def overall_hd95(self):
        if 'overall_hd95' not in self._cache:
            self._cache['overall_hd95'] = overall_hd(
                *self.surface_distances, percentile=95
            )
        return self._cache['overall_hd95']

    @property
    def overall_asd(self):
        if 'overall_asd' not in self._cache:
            self._cache['overall_asd'] = overall_asd(
                *self.surface_distances
            )
        return self._cache['overall_asd']

    @property
    def overall_msd(self):
        if 'overall_msd' not in self._cache:
            self._cache['overall_msd'] = overall_hd(
                *self.surface_distances, percentile=50
            )
        return self._cache['overall_msd']

    @property
    def structure_dice(self):
        if 'structure_dice' not in self._cache:
            self._cache['structure_dice'] = structure_dice(self.overlap, *self.areas, self.dice_threshold)
        return self._cache['structure_dice']

    @property
    def overall_dice(self):
        if 'overall_dice' not in self._cache:
            self._cache['overall_dice'] = overall_dice(self.masks[0], self.masks[1])
        return self._cache['overall_dice']

    @property
    def num_detected_objects(self):
        return num_detected_objects(*self.relative_overlap, self.detection_threshold)

    @property
    def coverage_fraction(self):
        if 'coverage_fraction' not in self._cache:
            self._cache['coverage_fraction'] = compute_coverage_fraction(*self.relative_overlap)
        return self._cache['coverage_fraction']

    @property
    def detected_mask(self):
        return (
            self.coverage_fraction[0] > self.detection_threshold,
            self.coverage_fraction[1] > self.detection_threshold,
        )

    def get_detected(self, metric):
        detected_metric_0 = {
            i: metric[0][i]
            for i, detected in enumerate(self.detected_mask[0])
            if detected
        }
        detected_metric_1 = {
            i: metric[1][i]
            for i, detected in enumerate(self.detected_mask[1])
            if detected
        }
        return detected_metric_0, detected_metric_1

    def get_undetected(self, metric):
        undetected_metric_0 = {
            i: metric[0][i]
            for i, detected in enumerate(self.detected_mask[0])
            if not detected
        }
        undetected_metric_1 = {
            i: metric[1][i]
            for i, detected in enumerate(self.detected_mask[1])
            if not detected
        }
        return undetected_metric_0, undetected_metric_1

    def __getattr__(self, name):
        if name.startswith("detected_"):
            name_idx = len("detected_")
            return self.get_detected(getattr(self, name[name_idx:]))
        elif name.startswith("undetected_"):
            name_idx = len("undetected_")
            return self.get_undetected(getattr(self, name[name_idx:]))
        else:
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

    @property
    def detected_metrics(self):
        return {
            f"detected_structure_dice_{self.names[0]}": list(self.detected_structure_dice[0].values()),
            f"detected_areas_{self.names[0]}": list(self.detected_areas[0].values()),
            f"detected_coverage_fraction_{self.names[0]}": list(
                self.detected_coverage_fraction[0].values()
            ),
            f"detected_structure_hd_{self.names[0]}_{self.names[1]}": list(
                self.detected_structure_hd[0].values()
            ),
            f"detected_structure_hd95_{self.names[0]}_{self.names[1]}": list(
                self.detected_structure_hd95[0].values()
            ),
            f"detected_structure_msd_{self.names[0]}_{self.names[1]}": list(
                self.detected_structure_msd[0].values()
            ),
            f"detected_structure_structure_asd_{self.names[0]}_{self.names[1]}": list(
                self.detected_structure_asd[0].values()
            ),
            f"detected_structure_dice_{self.names[1]}": list(self.detected_structure_dice[1].values()),
            f"detected_areas_{self.names[1]}": list(self.detected_areas[1].values()),
            f"detected_coverage_fraction_{self.names[1]}": list(
                self.detected_coverage_fraction[1].values()
            ),
            f"detected_structure_hd_{self.names[1]}_{self.names[0]}": list(
                self.detected_structure_hd[1].values()
            ),
            f"detected_structure_hd95_{self.names[1]}_{self.names[0]}": list(
                self.detected_structure_hd95[1].values()
            ),
            f"detected_structure_msd_{self.names[1]}_{self.names[0]}": list(
                self.detected_structure_msd[1].values()
            ),
            f"detected_structure_asd_{self.names[1]}_{self.names[0]}": list(
                self.detected_structure_asd[1].values()
            ),
        }

    @property
    def undetected_metrics(self):
        return {
            f"undetected_structure_dice_{self.names[0]}": list(
                self.undetected_structure_dice[0].values()
            ),
            f"undetected_areas_{self.names[0]}": list(self.undetected_areas[0].values()),
            f"undetected_coverage_fraction_{self.names[0]}": list(
                self.undetected_coverage_fraction[0].values()
            ),
            f"undetected_structure_hd_{self.names[0]}_{self.names[1]}": list(
                self.undetected_structure_hd[0].values()
            ),
            f"undetected_structure_hd95_{self.names[0]}_{self.names[1]}": list(
                self.undetected_structure_hd95[0].values()
            ),
            f"undetected_structure_msd_{self.names[0]}_{self.names[1]}": list(
                self.undetected_structure_msd[0].values()
            ),
            f"undetected_structure_asd_{self.names[0]}_{self.names[1]}": list(
                self.undetected_structure_asd[0].values()
            ),
            f"undetected_structure_dice_{self.names[1]}": list(self.undetected_structure_dice[1].values()),
            f"undetected_areas_{self.names[1]}": list(self.undetected_areas[1].values()),
            f"undetected_coverage_fraction_{self.names[1]}": list(
                self.undetected_coverage_fraction[1].values()
            ),
            f"undetected_structure_hd_{self.names[1]}_{self.names[0]}": list(
                self.undetected_structure_hd[1].values()
            ),
            f"undetected_structure_hd95_{self.names[1]}_{self.names[0]}": list(
                self.undetected_structure_hd95[1].values()
            ),
            f"undetected_structure_msd_{self.names[1]}_{self.names[0]}": list(
                self.undetected_structure_msd[1].values()
            ),
            f"undetected_structure_asd_{self.names[1]}_{self.names[0]}": list(
                self.undetected_structure_asd[1].values()
            ),
        }

    def _summary_statistics_from_dict(self, metric):
        return {
            f"{metric}_{self.names[0]}_{self.names[1]}_mean": np.mean(
                list(getattr(self, metric)[0].values())
            ),
            f"{metric}_{self.names[1]}_{self.names[0]}_mean": np.mean(
                list(getattr(self, metric)[1].values())
            ),
            f"{metric}_{self.names[0]}_{self.names[1]}_std": np.std(
                list(getattr(self, metric)[0].values())
            ),
            f"{metric}_{self.names[1]}_{self.names[0]}_std": np.std(
                list(getattr(self, metric)[1].values())
            ),
        }

    @property
    def summary_statistics(self):
        return {
            f"overall_hd_{self.names[0]}_{self.names[1]}": self.overall_hd[0],
            f"overall_hd95_{self.names[0]}_{self.names[1]}": self.overall_hd95[0],
            f"overall_asd{self.names[0]}_{self.names[1]}": self.overall_asd[0],
            f"overall_msd{self.names[0]}_{self.names[1]}": self.overall_msd[0],
            f"overall_hd_{self.names[1]}_{self.names[0]}": self.overall_hd[1],
            f"overall_hd95_{self.names[1]}_{self.names[0]}": self.overall_hd95[1],
            f"overall_asd{self.names[1]}_{self.names[0]}": self.overall_asd[1],
            f"overall_msd{self.names[1]}_{self.names[0]}": self.overall_msd[1],
            f"overall_dice": self.overall_dice,
            f"num_objects_{self.names[0]}": self.num_labels[0],
            f"num_objects_{self.names[1]}": self.num_labels[1],
            f"num_detected_objects_{self.names[0]}": self.num_detected_objects[0],
            f"num_detected_objects_{self.names[1]}": self.num_detected_objects[1],
            f"coverage_fraction_{self.names[0]}_mean": self.coverage_fraction[0].mean(),
            f"coverage_fraction_{self.names[1]}_mean": self.coverage_fraction[1].mean(),
            f"coverage_fraction_{self.names[0]}_std": self.coverage_fraction[0].std(),
            f"coverage_fraction_{self.names[1]}_std": self.coverage_fraction[1].std(),
            **self._summary_statistics_from_dict("detected_coverage_fraction"),
            **self._summary_statistics_from_dict("detected_structure_hd"),
            **self._summary_statistics_from_dict("detected_structure_hd95"),
            **self._summary_statistics_from_dict("detected_structure_msd"),
            **self._summary_statistics_from_dict("detected_structure_asd"),
            **self._summary_statistics_from_dict("detected_structure_dice"),
            **self._summary_statistics_from_dict("detected_areas"),
            **self._summary_statistics_from_dict("undetected_coverage_fraction"),
            **self._summary_statistics_from_dict("undetected_structure_hd"),
            **self._summary_statistics_from_dict("undetected_structure_hd95"),
            **self._summary_statistics_from_dict("undetected_structure_msd"),
            **self._summary_statistics_from_dict("undetected_structure_asd"),
            **self._summary_statistics_from_dict("undetected_structure_dice"),
            **self._summary_statistics_from_dict("undetected_areas"),
        }

    def evaluate(self, mask1, mask2):
        self.masks = (mask1, mask2)
        return EvaluationOutput(
            self.summary_statistics, self.detected_metrics, self.undetected_metrics
        )


class MasksEvaluator:
    def __init__(self, progress=True, *args, **kwargs):
        """All arguments except progress are passed on to the constructor of a MaskEvaluator.
        """
        self.progress = progress
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, masks1, masks2):
        self.summaries = defaultdict(list)
        self.evaluators = []
        self.detected_metrics = defaultdict(list)
        self.undetected_metrics = defaultdict(list)
        for mask1, mask2 in zip(tqdm(masks1), masks2):
            evaluator = MaskEvaluator(*self.args, **self.kwargs)
            self.evaluators.append(evaluator)
            evaluator.evaluate(mask1, mask2)
            for metric, value in evaluator.summary_statistics.items():
                self.summaries[metric].append(value)

            for metric, values in evaluator.detected_metrics.items():
                self.detected_metrics[metric] += list(values)

            for metric, values in evaluator.undetected_metrics.items():
                self.undetected_metrics[metric] += list(values)

        return EvaluationOutput(
            self.summaries, self.detected_metrics, self.undetected_metrics
        )


if __name__ == "__main__":
    targets = {
        id: experiment_results["PET"]["target"][slices].squeeze()
        for id, slices in patient_ids.items()
    }
    preds = {
        id: experiment_results["PET"]["prediction"][slices].squeeze()
        for id, slices in patient_ids.items()
    }

    target_labels = label(target)[0]
    target_objects = find_objects(target_labels)
    pred_labels = label(pred)[0]
    pred_objects = find_objects(pred_labels)

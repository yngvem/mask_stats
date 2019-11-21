from collections import defaultdict
from itertools import product
from numbers import Number
from typing import Dict, List, NamedTuple

import numpy as np
from scipy.ndimage import find_objects, label, morphological_gradient
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm


__all__ = [EvaluationOutput, MaskEvaluator, MasksEvaluator]


def label_overlap(labelled_1, labelled_2, label_1, label_2):
    """Compute overlap between `label_1` and `label_2`.
    """
    labelled_1 = labelled_1 == label_1
    labelled_2 = labelled_2 == label_2
    return np.sum(labelled_1 * labelled_2)


def measure_overlap(labelled_1, labelled_2, num_labels_1, num_labels_2, locs_1, locs_2):
    """Compute the overlap between two region labelled masks.
    """
    overlap = np.zeros((num_labels_1, num_labels_2))

    # Generate iterators
    info_1 = enumerate(locs_1)
    info_2 = enumerate(locs_2)

    for info_1, info_2 in product(info_1, info_2):
        # Extract info from iterators
        idx_1, location_1 = info_1
        idx_2, location_2 = info_2

        # Compute overlap
        overlap[idx_1, idx_2] = label_overlap(
            labelled_1[location_1], labelled_2[location_1], idx_1 + 1, idx_2 + 1
        )
    return overlap


def overlapping_areas(labelled_1, labelled_2, overlaps):
    """Return a dictionary that maps each label to the overlapping labels in the 
    """
    overlapping_areas_1 = {
        i + 1: [j + 1 for j, overlap in enumerate(overlap_vector) if overlap != 0]
        for i, overlap_vector in enumerate(overlaps)
    }

    overlapping_areas_2 = {
        i + 1: [j + 1 for j, overlap in enumerate(overlap_vector) if overlap != 0]
        for i, overlap_vector in enumerate(overlaps.T)
    }

    return overlapping_areas_1, overlapping_areas_2


def compute_areas(labelled, num_labels):
    return np.array([np.sum(labelled == label) for label in range(1, num_labels + 1)])


def relative_overlap(areas_1, areas_2, overlap):
    return (overlap / areas_1.reshape(-1, 1), overlap / areas_2.reshape(1, -1))


def total_relative_overlap(relative_1_overlap, relative_2_overlap):
    return relative_overlap_1.sum(1), relative_overlap_2.sum(0)


def _norm(x):
    return np.sqrt(np.sum(x ** 2, axis=x.ndim - 1))


def _set_distances(nonzeros_1, nonzeros_2):
    """Compute the percentile Husdorff distance between two sets.
    """
    distances = np.zeros(len(nonzeros_1))
    for i, _ in enumerate(distances):
        distances[i] = np.min(_norm(nonzeros_1[i].reshape(1, -1) - nonzeros_2))
    # distances = np.linalg.norm(
    #        nonzeros_1[:, np.newaxis] - nonzeros_2[np.newaxis], axis=-1
    # )
    return distances


def percentile_directed_hausdorff(nonzeros_1, nonzeros_2, percentile):
    distances = _directed_distances(nonzeros_1, nonzeros_2)
    return np.percentile(distances, percentile)


def surface_distances(mask1, mask2):
    """Return the surface distances for all points on the surface of mask1 to the surface of mask2.
    """
    structuring_el_size = tuple(3 for _ in mask1.shape)
    grad1 = morphological_gradient(mask1.astype(int), size=structuring_el_size)
    grad2 = morphological_gradient(mask2.astype(int), size=structuring_el_size)

    nonzeros_1 = np.array(np.nonzero(grad1)).T
    nonzeros_2 = np.array(np.nonzero(grad2)).T
    return _set_distances(nonzeros_1, nonzeros_2)


def labelled_surface_distances(labelled_1, labelled_2, num_labels_1, num_labels_2):
    """Compute the surface distances for for all connected components in one mask to the whole second mask.
    """
    mask1 = labelled_1 != 0
    mask2 = labelled_2 != 0

    surface_distance_label_1 = []
    for idx in range(num_labels_1):
        surface_distance_label_1.append(surface_distances(labelled_1 == idx + 1, mask2))

    surface_distance_label_2 = []
    for idx in range(num_labels_2):
        surface_distance_label_2.append(surface_distances(labelled_2 == idx + 1, mask1))

    return surface_distance_label_1, surface_distance_label_2


def labelled_hausdorff(
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


def labelled_average_surface_distance(
    labelled_surface_distances_1, labelled_surface_distances_2
):
    """Compute the Hausdorff distance for for all connected components in one mask to the whole second mask.
    """
    average_surface_distance_label_1 = []
    for surface_distance in labelled_surface_distances_1:
        average_surface_distance_label_1.append(np.mean(surface_distance))

    average_surface_distance_label_2 = []
    for surface_distance in labelled_surface_distances_2:
        average_surface_distance_label_2.append(np.mean(surface_distance))

    return (
        np.array(average_surface_distance_label_1),
        np.array(average_surface_distance_label_2),
    )


def labelled_dice(overlap, areas_1, areas_2, threshold=0):
    """Compute the component-wise dice.

    For each connected component, C, in the first mask, we find the connected
    components in the second mask that covers a factor of at least `threshold`
    times the area of C. Once we have these components, we compute the dice.
    """

    labelled_dice_1 = []
    for i, area in enumerate(areas_1):
        relevant_areas_2 = 2*overlap[i] / area > threshold
        labelled_dice_1.append(
            2*overlap[i].sum() / (area + areas_2[relevant_areas_2].sum())
        )

    labelled_dice_2 = []
    for i, area in enumerate(areas_2):
        relevant_areas_1 = overlap[:, i] / area > threshold
        labelled_dice_2.append(
            2*overlap[:, i].sum() / (area + areas_1[relevant_areas_1].sum())
        )

    return labelled_dice_1, labelled_dice_2


def num_detected_objects(relative_overlap_1, relative_overlap_2, threshold):
    detected_object_1 = np.sum(relative_overlap_1.sum(1) > threshold)
    detected_object_2 = np.sum(relative_overlap_2.sum(0) > threshold)
    return detected_object_1, detected_object_2


def coverage_fraction(relative_overlap_1, relative_overlap_2):
    return relative_overlap_1.sum(1), relative_overlap_2.sum(0)


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
        self._should_recompute = {
            "labelled": True,
            "locations": True,
            "overlap": True,
            "areas": True,
            "relative_overlap": True,
            "total_relative_overlap": True,
            "hausdorff": True,
            "hausdorff_95": True,
            "average_surface_distance": True,
            "median_surface_distance": True,
            "surface_distances": True,
            "dice": True,
            "coverage_fraction": True,
            "num_detected_objects": True,
        }

    @property
    def detection_threshold(self):
        self._should_recompute["num_detected_objects"] = True
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
        self._should_recompute = {measure: True for measure in self._should_recompute}

    @property
    def masks(self):
        return self._masks

    @masks.setter
    def masks(self, value):
        self._reset_cache()
        self._masks = tuple(value)

    @property
    def labelled(self):
        if self._should_recompute["labelled"]:
            labelled0, num_labels0 = label(self.masks[0])
            labelled1, num_labels1 = label(self.masks[1])
            self._labelled = (labelled0, labelled1)
            self._num_labels = (num_labels0, num_labels1)
            self._should_recompute["labelled"] = False
        return self._labelled

    @property
    def num_labels(self):
        if self._should_recompute["labelled"]:
            labelled0, num_labels0 = label(self.masks[0])
            labelled1, num_labels1 = label(self.masks[1])
            self._labelled = (labelled0, labelled1)
            self._num_labels = (num_labels0, num_labels1)
            self._should_recompute["labelled"] = False
        return self._num_labels

    @property
    def locations(self):
        if self._should_recompute["locations"] or self._should_recompute["labelled"]:
            self._locations = tuple(
                find_objects(labelled) for labelled in self.labelled
            )
            self._should_recompute["locations"] = False
        return self._locations

    @property
    def overlap(self):
        if (
            self._should_recompute["overlap"]
            or self._should_recompute["locations"]
            or self._should_recompute["labelled"]
        ):
            self._overlap = measure_overlap(
                *self.labelled, *self.num_labels, *self.locations
            )
            self._should_recompute["overlap"] = False
        return self._overlap

    @property
    def areas(self):
        if self._should_recompute["areas"] or self._should_recompute["labelled"]:
            self._areas = (
                compute_areas(self.labelled[0], self.num_labels[0]),
                compute_areas(self.labelled[1], self.num_labels[1]),
            )
            self._should_recompute["areas"] = False
        return self._areas

    @property
    def relative_overlap(self):
        if (
            self._should_recompute["relative_overlap"]
            or self._should_recompute["overlap"]
            or self._should_recompute["areas"]
        ):
            self._relative_overlap = relative_overlap(*self.areas, self.overlap)
            self._should_recompute["relative_overlap"] = False
        return self._relative_overlap

    @property
    def total_relative_overlap(self):
        if (
            self._should_recompute["total_relative_overlap"]
            or self._should_recompute["relative_overlap"]
        ):
            self._total_relative_overlap = total_relative_overlap(
                *self.relative_overlap
            )
            self._should_recompute["total_relative_overlap"] = False
        return self._total_relative_overlap

    @property
    def surface_distances(self):
        if (
            self._should_recompute["surface_distances"]
            or self._should_recompute["labelled"]
        ):
            self._surface_distances = labelled_surface_distances(
                *self.labelled, *self.num_labels
            )
            self._should_recompute["surface_distances"] = False
        return self._surface_distances

    @property
    def hausdorff(self):
        if (
            self._should_recompute["hausdorff"]
            or self._should_recompute["surface_distances"]
        ):
            self._hausdorff = labelled_hausdorff(
                *self.surface_distances, percentile=100
            )
            self._should_recompute["hausdorff"] = False
        return self._hausdorff

    @property
    def hausdorff_95(self):
        if (
            self._should_recompute["hausdorff_95"]
            or self._should_recompute["surface_distances"]
        ):
            self._hausdorff_95 = labelled_hausdorff(
                *self.surface_distances, percentile=95
            )
            self._should_recompute["hausdorff_95"] = False
        return self._hausdorff_95

    @property
    def average_surface_distance(self):
        if (
            self._should_recompute["average_surface_distance"]
            or self._should_recompute["surface_distances"]
        ):
            self._average_surface_distance = labelled_average_surface_distance(
                *self.surface_distances
            )
            self._should_recompute["average_surface_distance"] = False
        return self._average_surface_distance

    @property
    def median_surface_distance(self):
        if (
            self._should_recompute["median_surface_distance"]
            or self._should_recompute["surface_distances"]
        ):
            self._median_surface_distance = labelled_hausdorff(
                *self.surface_distances, percentile=50
            )
            self._should_recompute["median_surface_distance"] = False
        return self._median_surface_distance

    @property
    def dice(self):
        if (
            self._should_recompute["dice"]
            or self._should_recompute["overlap"]
            or self._should_recompute["areas"]
        ):
            self._dice = labelled_dice(self.overlap, *self.areas, self.dice_threshold)
            self._should_recompute["dice"] = False
        return self._dice

    @property
    def num_detected_objects(self):
        return num_detected_objects(*self.relative_overlap, self.detection_threshold)

    @property
    def coverage_fraction(self):
        if (
            self._should_recompute["coverage_fraction"]
            or self._should_recompute["relative_overlap"]
        ):
            self._coverage_fraction = coverage_fraction(*self.relative_overlap)
            self._should_recompute["coverage_fraction"] = False
        return self._coverage_fraction

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
            f"detected_dice_{self.names[0]}": list(self.detected_dice[0].values()),
            f"detected_areas_{self.names[0]}": list(self.detected_areas[0].values()),
            f"detected_coverage_fraction_{self.names[0]}": list(
                self.detected_coverage_fraction[0].values()
            ),
            f"detected_hausdorff_{self.names[0]}_{self.names[1]}": list(
                self.detected_hausdorff[0].values()
            ),
            f"detected_hausdorff_95_{self.names[0]}_{self.names[1]}": list(
                self.detected_hausdorff_95[0].values()
            ),
            f"detected_median_surface_distance_{self.names[0]}_{self.names[1]}": list(
                self.detected_median_surface_distance[0].values()
            ),
            f"detected_average_surface_distance_{self.names[0]}_{self.names[1]}": list(
                self.detected_average_surface_distance[0].values()
            ),
            f"detected_dice_{self.names[1]}": list(self.detected_dice[1].values()),
            f"detected_areas_{self.names[1]}": list(self.detected_areas[1].values()),
            f"detected_coverage_fraction_{self.names[1]}": list(
                self.detected_coverage_fraction[1].values()
            ),
            f"detected_hausdorff_{self.names[1]}_{self.names[0]}": list(
                self.detected_hausdorff[1].values()
            ),
            f"detected_hausdorff_95_{self.names[1]}_{self.names[0]}": list(
                self.detected_hausdorff_95[1].values()
            ),
            f"detected_median_surface_distance_{self.names[1]}_{self.names[0]}": list(
                self.detected_median_surface_distance[1].values()
            ),
            f"detected_average_surface_distance_{self.names[1]}_{self.names[0]}": list(
                self.detected_average_surface_distance[1].values()
            ),
        }

    @property
    def undetected_metrics(self):
        return {
            f"undetected_coverage_fraction_{self.names[0]}": list(
                self.undetected_coverage_fraction[0].values()
            ),
            f"undetected_areas_{self.names[0]}": list(
                self.undetected_areas[0].values()
            ),
            f"undetected_coverage_fraction_{self.names[1]}": list(
                self.undetected_coverage_fraction[1].values()
            ),
            f"undetected_areas_{self.names[1]}": list(
                self.undetected_areas[1].values()
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
            f"num_objects_{self.names[0]}": self.num_labels[0],
            f"num_objects_{self.names[1]}": self.num_labels[1],
            f"num_detected_objects_{self.names[0]}": self.num_detected_objects[0],
            f"num_detected_objects_{self.names[1]}": self.num_detected_objects[1],
            f"coverage_fraction_{self.names[0]}_mean": self.coverage_fraction[0].mean(),
            f"coverage_fraction_{self.names[1]}_mean": self.coverage_fraction[1].mean(),
            f"coverage_fraction_{self.names[0]}_std": self.coverage_fraction[0].std(),
            f"coverage_fraction_{self.names[1]}_std": self.coverage_fraction[1].std(),
            **self._summary_statistics_from_dict("detected_coverage_fraction"),
            **self._summary_statistics_from_dict("detected_hausdorff"),
            **self._summary_statistics_from_dict("detected_hausdorff_95"),
            **self._summary_statistics_from_dict("detected_median_surface_distance"),
            **self._summary_statistics_from_dict("detected_average_surface_distance"),
            **self._summary_statistics_from_dict("detected_dice"),
            **self._summary_statistics_from_dict("detected_areas"),
            **self._summary_statistics_from_dict("undetected_dice"),
            **self._summary_statistics_from_dict("undetected_areas"),
        }

    def evaluate(self, mask1, mask2):
        self.masks = (mask1, mask2)
        return EvaluationOutput(self.summary_statistics, self.detected_metrics, self.undetected_metrics)


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

        return EvaluationOutput(self.summaries, self.detected_metrics, self.undetected_metrics)


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

from collections import defaultdict
from itertools import product

import numpy as np
from scipy.ndimage import find_objects, label, morphological_gradient
from scipy.spatial.distance import directed_hausdorff


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


def boundary_hausdorff(mask1, mask2):
    """Compute the Hausdorff distance between the boundary of two masks.
    """
    structuring_el_size = tuple(3 for _ in mask1.shape)
    grad1 = morphological_gradient(mask1.astype(int), size=structuring_el_size)
    grad2 = morphological_gradient(mask2.astype(int), size=structuring_el_size)

    nonzeros_1 = np.array(np.nonzero(grad1)).T
    nonzeros_2 = np.array(np.nonzero(grad2)).T

    return directed_hausdorff(nonzeros_1, nonzeros_2)[0]


def labelled_hausdorff(labelled_1, labelled_2, num_labels_1, num_labels_2):
    """Compute the Hausdorff distance for for all connected components in one mask to the whole second mask.
    """
    mask1 = labelled_1 != 0
    mask2 = labelled_2 != 0

    hausdorffs_label_1 = []
    for idx in range(num_labels_1):
        hausdorffs_label_1.append(boundary_hausdorff(labelled_1 == idx + 1, mask2))

    hausdorffs_label_2 = []
    for idx in range(num_labels_1):
        hausdorffs_label_2.append(boundary_hausdorff(labelled_2 == idx + 1, mask1))

    return np.array(hausdorffs_label_1), np.array(hausdorffs_label_2)


def labelled_dice(overlap, areas_1, areas_2, threshold=0):
    """Compute the component-wise dice.

    For each connected component, C, in the first mask, we find the connected
    components in the second mask that covers a factor of at least `threshold`
    times the area of C. Once we have these components, we compute the dice.
    """

    labelled_dice_1 = []
    for i, area in enumerate(areas_1):
        relevant_areas_2 = overlap[i] / area > threshold
        labelled_dice_1.append(
            overlap[i].sum() / (area + areas_2[relevant_areas_2].sum())
        )

    labelled_dice_2 = []
    for i, area in enumerate(areas_2):
        relevant_areas_1 = overlap[:, i] / area > threshold
        labelled_dice_2.append(
            overlap[:, i].sum() / (area + areas_1[relevant_areas_1].sum())
        )

    return labelled_dice_1, labelled_dice_2


def num_detected_objects(relative_overlap_1, relative_overlap_2, threshold):
    detected_object_1 = np.sum(relative_overlap_1.sum(1) > threshold)
    detected_object_2 = np.sum(relative_overlap_2.sum(0) > threshold)
    return detected_object_1, detected_object_2


def coverage_fraction(relative_overlap_1, relative_overlap_2):
    return relative_overlap_1.sum(1), relative_overlap_2.sum(0)


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

        self.dice_threshold = dice_threshold
        self.detection_threshold = detection_threshold
        self._should_recompute = {
            "labelled": True,
            "locations": True,
            "overlap": True,
            "areas": True,
            "relative_overlap": True,
            "total_relative_overlap": True,
            "labelled_hausdorff": True,
            "labelled_dice": True,
            "coverage_fraction": True,
            "num_detected_objects": True,
        }

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
        if self._should_recompute["locations"]:
            self._locations = tuple(
                find_objects(labelled) for labelled in self.labelled
            )
            self._should_recompute["locations"] = False
        return self._locations

    @property
    def overlap(self):
        if self._should_recompute["overlap"]:
            self._overlap = measure_overlap(
                *self.labelled, *self.num_labels, *self.locations
            )
            self._should_recompute["overlap"] = False
        return self._overlap

    @property
    def areas(self):
        if self._should_recompute["areas"]:
            self._areas = (
                compute_areas(self.labelled[0], self.num_labels[0]),
                compute_areas(self.labelled[1], self.num_labels[1]),
            )
            self._should_recompute["areas"] = False
        return self._areas

    @property
    def relative_overlap(self):
        if self._should_recompute["relative_overlap"]:
            self._relative_overlap = relative_overlap(*self.areas, self.overlap)
            self._should_recompute["relative_overlap"] = False
        return self._relative_overlap

    @property
    def total_relative_overlap(self):
        if self._should_recompute["total_relative_overlap"]:
            self._total_relative_overlap = total_relative_overlap(
                *self.relative_overlap
            )
            self._should_recompute["total_relative_overlap"] = False
        return self._total_relative_overlap

    @property
    def labelled_hausdorff(self):
        if self._should_recompute["labelled_hausdorff"]:
            self._labelled_hausdorff = labelled_hausdorff(
                *self.labelled, *self.num_labels
            )
            self._should_recompute["labelled_hausdorff"] = False
        return self._labelled_hausdorff

    @property
    def labelled_dice(self):
        if self._should_recompute["labelled_dice"]:
            self._labelled_dice = labelled_dice(
                self.overlap, *self.areas, self.dice_threshold
            )
            self._should_recompute["labelled_dice"] = False
        return self._labelled_dice

    @property
    def num_detected_objects(self):
        if self._should_recompute["num_detected_objects"]:
            self._num_detected_objects = num_detected_objects(
                *self.relative_overlap, self.detection_threshold
            )
            self._should_recompute["num_detected_objects"] = False
        return self._num_detected_objects

    @property
    def coverage_fraction(self):
        if self._should_recompute["coverage_fraction"]:
            self._coverage_fraction = coverage_fraction(*self.relative_overlap)
            self._should_recompute["coverage_fraction"] = False
        return self._coverage_fraction

    @property
    def detected_mask(self):
        return (
            self.coverage_fraction[0] > self.detection_threshold,
            self.coverage_fraction[1] > self.detection_threshold,
        )

    @property
    def detected_hausdorff(self):
        detected_hausdorff_0 = {
            i: self.labelled_hausdorff[0][i]
            for i, detected in enumerate(self.detected_mask[0])
            if detected
        }
        detected_hausdorff_1 = {
            i: self.labelled_hausdorff[1][i]
            for i, detected in enumerate(self.detected_mask[1])
            if detected
        }
        return detected_hausdorff_0, detected_hausdorff_1

    @property
    def detected_dice(self):
        detected_dice_0 = {
            i: self.labelled_dice[0][i]
            for i, detected in enumerate(self.detected_mask[0])
            if detected
        }
        detected_dice_1 = {
            i: self.labelled_dice[1][i]
            for i, detected in enumerate(self.detected_mask[1])
            if detected
        }
        return detected_dice_0, detected_dice_1

    @property
    def detected_areas(self):
        detected_areas_0 = {
            i: self.areas[0][i]
            for i, detected in enumerate(self.detected_mask[0])
            if detected
        }
        detected_areas_1 = {
            i: self.areas[1][i]
            for i, detected in enumerate(self.detected_mask[1])
            if detected
        }
        return detected_areas_0, detected_areas_1

    @property
    def undetected_areas(self):
        undetected_areas_0 = {
            i: self.areas[0][i]
            for i, detected in enumerate(self.detected_mask[0])
            if not detected
        }
        undetected_areas_1 = {
            i: self.areas[1][i]
            for i, detected in enumerate(self.detected_mask[1])
            if not detected
        }
        return undetected_areas_0, undetected_areas_1

    def evaluate(self, mask1, mask2):
        self.masks = (mask1, mask2)
        self.summary_statistics = {
            f"num_objects_{self.names[0]}": self.num_labels[0],
            f"num_objects_{self.names[1]}": self.num_labels[1],
            f"num_detected_objects_{self.names[0]}": self.num_detected_objects[0],
            f"num_detected_objects_{self.names[1]}": self.num_detected_objects[1],
            f"coverage_fraction_{self.names[0]}_mean": self.coverage_fraction[0].mean(),
            f"coverage_fraction_{self.names[1]}_mean": self.coverage_fraction[1].mean(),
            f"coverage_fraction_{self.names[0]}_std": self.coverage_fraction[0].std(),
            f"coverage_fraction_{self.names[1]}_std": self.coverage_fraction[1].std(),
            f"detected_hausdorff_{self.names[0]}_{self.names[1]}_mean": np.mean(
                list(self.detected_hausdorff[0].values())
            ),
            f"detected_hausdorff_{self.names[1]}_{self.names[0]}_mean": np.mean(
                list(self.detected_hausdorff[1].values())
            ),
            f"detected_hausdorff_{self.names[0]}_{self.names[1]}_std": np.std(
                list(self.detected_hausdorff[0].values())
            ),
            f"detected_hausdorff_{self.names[1]}_{self.names[0]}_std": np.std(
                list(self.detected_hausdorff[1].values())
            ),
            f"detected_dice_{self.names[0]}_{self.names[1]}_mean": np.mean(
                list(self.detected_dice[0].values())
            ),
            f"detected_dice_{self.names[1]}_{self.names[0]}_mean": np.mean(
                list(self.detected_dice[1].values())
            ),
            f"detected_dice_{self.names[0]}_{self.names[1]}_std": np.std(
                list(self.detected_dice[0].values())
            ),
            f"detected_dice_{self.names[1]}_{self.names[0]}_std": np.std(
                list(self.detected_dice[1].values())
            ),
            f"detected_area_mean_{self.names[0]}": np.mean(
                list(self.detected_areas[0].values())
            ),
            f"detected_area_mean_{self.names[1]}": np.mean(
                list(self.detected_areas[1].values())
            ),
            f"detected_area_std_{self.names[0]}": np.std(
                list(self.detected_areas[0].values())
            ),
            f"detected_area_std_{self.names[1]}": np.std(
                list(self.detected_areas[1].values())
            ),
            f"undetected_area_mean_{self.names[0]}": np.mean(
                list(self.undetected_areas[0].values())
            ),
            f"undetected_area_mean_{self.names[1]}": np.mean(
                list(self.undetected_areas[1].values())
            ),
            f"undetected_area_std_{self.names[0]}": np.std(
                list(self.undetected_areas[0].values())
            ),
            f"undetected_area_std_{self.names[1]}": np.std(
                list(self.undetected_areas[1].values())
            ),
        }
        self.metrics = {
            f"detected_hausdorff_{self.names[0]}_{self.names[1]}": self.detected_hausdorff[
                0
            ],
            f"detected_hausdorff_{self.names[1]}_{self.names[0]}": self.detected_hausdorff[
                1
            ],
            f"detected_dice_{self.names[0]}": self.detected_dice[0],
            f"detected_dice_{self.names[1]}": self.detected_dice[1],
            f"coverage_fraction_{self.names[0]}": self.coverage_fraction[0],
            f"coverage_fraction_{self.names[1]}": self.coverage_fraction[1],
            f"detected_areas_{self.names[0]}": self.detected_areas[0],
            f"detected_areas_{self.names[1]}": self.detected_areas[1],
            f"undetected_areas_{self.names[0]}": self.undetected_areas[0],
            f"undetected_areas_{self.names[1]}": self.undetected_areas[1],
        }
        return self.summary_statistics, self.metrics


class MasksEvaluator:
    def __init__(self, *args, **kwargs):
        """All arguments are passed on to the constructor of a MaskEvaluator.
        """
        self.evaluator = MaskEvaluator(*args, **kwargs)

    def evaluate(self, masks1, masks2):
        self.summaries = defaultdict(list)
        self.metrics = defaultdict(list)
        for mask1, mask2 in zip(masks1, masks2):
            self.evaluator.evaluate(mask1, mask2)
            for metric, value in self.evaluator.summary_statistics.items():
                self.summaries[metric].append(value)

            for metric, values in self.evaluator.metrics.items():
                self.metrics[metric] += list(values)

        return self.summaries, self.metrics


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

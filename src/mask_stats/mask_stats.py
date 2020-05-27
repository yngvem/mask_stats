from collections import defaultdict
from itertools import product
from numbers import Number
from typing import Dict, List, NamedTuple

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage.morphology import find_objects, label
from tqdm import tqdm

from .surface_distance import (
    compute_labelled_surface_distances,
    compute_structure_hd,
    compute_structure_asd,
    compute_overall_hd,
    compute_overall_asd,
)
from .overlap import (
    compute_areas,
    compute_coverage_fraction,
    compute_label_overlap,
    compute_num_detected_objects,
    compute_overall_dice,
    compute_pairwise_overlaps,
    compute_relative_overlap,
    compute_structure_accuracy,
    compute_structure_dice
)

__all__ = ["EvaluationOutput", "MaskEvaluator", "MasksEvaluator"]


class EvaluationOutput(NamedTuple):
    summary_statistics: Dict[str, List[Number]]
    detected_metrics: Dict[str, List[Number]]
    undetected_metrics: Dict[str, List[Number]]


class _MetricRegister:
    def __init__(self):
        self.overall_metrics = {}
        self.structure_metrics = {}

    def as_overall_metric(self, metric):
        self.overall_metrics[metric.__name__] = metric
        return metric

    def as_structure_metric(self, metric):
        self.structure_metrics[metric.__name__] = metric
        return metric


def compute_overlap_metrics(mask1, mask2, detection_threshold):
    labelled_mask1, num_labels_mask1 = label(mask1)
    labelled_mask2, num_labels_mask2 = label(mask1)

    object_locations_mask1 = find_objects(labelled_mask1)
    object_locations_mask2 = find_objects(labelled_mask2)

    areas_mask1 = compute_areas(labelled_mask1, object_locations_mask1),
    areas_mask2 = compute_areas(labelled_mask2, object_locations_mask2),

    pairwise_overlaps = compute_pairwise_overlaps(
        labelled_mask1, 
        labelled_mask2, 
        num_labels_mask1, 
        num_labels_mask2,
        object_locations_mask1,
        object_locations_mask2,
    )

    relative_pairwise_overlaps = compute_relative_overlap(
        areas_mask1, areas_mask2, pairwise_overlaps
    )



class MaskEvaluator:
    """Compute performance metrics between two binary masks.
    """

    _metric_register = _MetricRegister()

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

    # ----------------- Mask attributes used by metrics ------------------ #
    @property
    def labelled(self):
        if "labelled" not in self._cache:
            labelled0, num_labels0 = label(self.masks[0])
            labelled1, num_labels1 = label(self.masks[1])
            self._cache["labelled"] = (labelled0, labelled1)
            self._cache["num_labels"] = (num_labels0, num_labels1)
        return self._cache["labelled"]

    @property
    def num_labels(self):
        if "num_labels" not in self._cache:
            labelled0, num_labels0 = label(self.masks[0])
            labelled1, num_labels1 = label(self.masks[1])
            self._cache["labelled"] = (labelled0, labelled1)
            self._cache["num_labels"] = (num_labels0, num_labels1)
        return self._cache["num_labels"]

    @property
    def locations(self):
        if "location" not in self._cache:
            self._cache["locations"] = tuple(
                find_objects(labelled) for labelled in self.labelled
            )
        return self._cache["locations"]

#
    @property
    def overlap(self):
        if "overlap" not in self._cache:
            self._cache["overlap"] = compute_pairwise_overlaps(
                *self.labelled, *self.num_labels, *self.locations
            )
        return self._cache["overlap"]

    @property
    @_metric_register.as_structure_metric
    def areas(self):
        return (
            compute_areas(self.labelled[0], self.num_labels[0]),
            compute_areas(self.labelled[1], self.num_labels[1]),
        )

    @property
    @_metric_register.as_structure_metric
    def relative_overlap(self):
        return compute_relative_overlap(*self.areas, self.overlap)

    @property
    def surface_distances(self):
        if "surface_distances" not in self._cache:
            self._cache["surface_distances"] = compute_labelled_surface_distances(
                *self.labelled, *self.num_labels
            )
        return self._cache["surface_distances"]

    # ----------------- Metrics ------------------ #
    @property
    @_metric_register.as_structure_metric
    def structure_hd(self):
        return structure_hd(*self.surface_distances, percentile=100)

    @property
    @_metric_register.as_structure_metric
    def structure_hd95(self):
        return structure_hd(*self.surface_distances, percentile=95)

    @property
    @_metric_register.as_structure_metric
    def structure_asd(self):
        return structure_asd(*self.surface_distances)

    @property
    @_metric_register.as_structure_metric
    def structure_msd(self):
        return structure_hd(*self.surface_distances, percentile=50)

    @property
    @_metric_register.as_overall_metric
    def overall_hd(self):
        return overall_hd(*self.surface_distances, percentile=100)

    @property
    @_metric_register.as_overall_metric
    def overall_hd95(self):
        return overall_hd(*self.surface_distances, percentile=95)

    @property
    @_metric_register.as_overall_metric
    def overall_asd(self):
        return overall_asd(*self.surface_distances)

    @property
    @_metric_register.as_overall_metric
    def overall_msd(self):
        return overall_hd(*self.surface_distances, percentile=50)

    @property
    @_metric_register.as_structure_metric
    def structure_dice(self):
        return structure_dice(self.overlap, *self.areas, self.dice_threshold)

    @property
    @_metric_register.as_overall_metric
    def overall_dice(self):
        return overall_dice(self.masks[0], self.masks[1])

    @property
    @_metric_register.as_overall_metric
    def num_detected_objects(self):
        return num_detected_objects(*self.relative_overlap, self.detection_threshold)

    @property
    @_metric_register.as_structure_metric
    def coverage_fraction(self):
        return compute_coverage_fraction(*self.relative_overlap)

    # ----------------- Utility methods ------------------ #
    @property
    def is_detected_mask(self):
        return (
            self.coverage_fraction[0] > self.detection_threshold,
            self.coverage_fraction[1] > self.detection_threshold,
        )

    def get_detected(self, metric):
        detected_metric_0 = {
            i: metric[0][i]
            for i, detected in enumerate(self.is_detected_mask[0])
            if detected
        }
        detected_metric_1 = {
            i: metric[1][i]
            for i, detected in enumerate(self.is_detected_mask[1])
            if detected
        }
        return detected_metric_0, detected_metric_1

    def get_undetected(self, metric):
        undetected_metric_0 = {
            i: metric[0][i]
            for i, detected in enumerate(self.is_detected_mask[0])
            if not detected
        }
        undetected_metric_1 = {
            i: metric[1][i]
            for i, detected in enumerate(self.is_detected_mask[1])
            if not detected
        }
        return undetected_metric_0, undetected_metric_1

    def __getattr__(self, name):
        if name.startswith("detected_"):
            name_idx = len("detected_")
            return self.get_detected(getattr([name[name_idx:]]))
        elif name.startswith("undetected_"):
            name_idx = len("undetected_")
            return self.get_undetected(getattr([name[name_idx:]]))
        else:
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")

    @property
    def detected_metrics(self):
        metrics = {}
        for metric_name in self._metric_register.structure_metrics:
            current_metrics = getattr(f"detected_{metric_name}")

            metrics[f"detected_{metric_name}_{self.names[0]}"] = list(
                current_metrics[0].values()
            )
            metrics[f"detected_{metric_name}_{self.names[1]}"] = list(
                current_metrics[1].values()
            )

        return metrics
        return {
            f"detected_structure_dice_{self.names[0]}": list(
                self.get_detected_structure_dice[0].values()
            ),
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
            f"detected_structure_dice_{self.names[1]}": list(
                self.detected_structure_dice[1].values()
            ),
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
        metrics = {}
        for metric_name in self._metric_register.structure_metrics:
            current_metrics = getattr(f"undetected_{metric_name}")

            metrics[f"undetected_{metric_name}_{self.names[0]}"] = list(
                current_metrics[0].values()
            )
            metrics[f"undetected_{metric_name}_{self.names[1]}"] = list(
                current_metrics[1].values()
            )

        return metrics
        return {
            f"undetected_structure_dice_{self.names[0]}": list(
                self.undetected_structure_dice[0].values()
            ),
            f"undetected_areas_{self.names[0]}": list(
                self.undetected_areas[0].values()
            ),
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
            f"undetected_structure_dice_{self.names[1]}": list(
                self.undetected_structure_dice[1].values()
            ),
            f"undetected_areas_{self.names[1]}": list(
                self.undetected_areas[1].values()
            ),
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

    def get_summary_statistics(self):
        return {
            f"overall_hd_{self.names[0]}_{self.names[1]}": self.overall_hd[0],
            f"overall_hd95_{self.names[0]}_{self.names[1]}": self.overall_hd95[0],
            f"overall_asd_{self.names[0]}_{self.names[1]}": self.overall_asd[0],
            f"overall_msd_{self.names[0]}_{self.names[1]}": self.overall_msd[0],
            f"overall_hd_{self.names[1]}_{self.names[0]}": self.overall_hd[1],
            f"overall_hd95_{self.names[1]}_{self.names[0]}": self.overall_hd95[1],
            f"overall_asd_{self.names[1]}_{self.names[0]}": self.overall_asd[1],
            f"overall_msd_{self.names[1]}_{self.names[0]}": self.overall_msd[1],
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
            self.get_summary_statistics(),
            self.detected_metrics,
            self.undetected_metrics,
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


def evaluate_all_masks(masks1, masks2, progress=True, *args, **kwargs):
    """Evaluate all input masks.
    """
    mask_summaries = defaultdict(list)
    all_detected_metrics = defaultdict(list)
    all_undetected_metrics = defaultdict(list)

    for mask1, mask2 in zip(tqdm(masks1), masks2):
        evaluator = MaskEvaluator(*args, **kwargs)
        for metric, value in evaluator.summary_statistics.items():
            summaries[metric].append(value)

        for metric, values in evaluator.detected_metrics.items():
            detected_metrics[metric] += list(values)

        for metric, values in evaluator.undetected_metrics.items():
            undetected_metrics[metric] += list(values)

    return EvaluationOutput(mask_summaries, detected_metrics, undetected_metrics)


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

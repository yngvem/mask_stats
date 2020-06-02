from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, label
from tqdm import tqdm

from ._overlap import (compute_coverage_fraction, compute_label_overlap,
                      compute_num_detected_objects, compute_overall_dice,
                      compute_pairwise_overlaps, compute_relative_overlaps,
                      compute_object_accuracy, compute_object_dice,
                      compute_volumes)
from ._surface_distance import (compute_labelled_surface_distances,
                               compute_overall_average_surface_distances,
                               compute_overall_percentile_surface_distances,
                               compute_object_average_surface_distances,
                               compute_object_percentile_surface_distances)


class MaskFeatures:
    def __init__(self, mask):
        self.mask = mask
        self.region_labelled_mask, self.num_objects = label(mask)
        self.object_locations = find_objects(self.region_labelled_mask)
        self.volume = compute_volumes(self.region_labelled_mask, self.num_objects)


class MaskPairFeatures:
    def __init__(self, mask1, mask2):
        self.mask1_features = MaskFeatures(mask1)
        self.mask2_features = MaskFeatures(mask2)
        self.pairwise_overlap_matrix = compute_pairwise_overlaps(
            self.mask1_features.region_labelled_mask, 
            self.mask2_features.region_labelled_mask, 
            self.mask1_features.num_objects, 
            self.mask2_features.num_objects,
            self.mask1_features.object_locations,
            self.mask2_features.object_locations,
        )

        out = compute_relative_overlaps(
            self.mask1_features.volume,
            self.mask2_features.volume,
            self.pairwise_overlap_matrix,
        )
        self.mask1_relative_overlaps = out[0]
        self.mask2_relative_overlaps = out[1]

        # Surface distances
        out = compute_labelled_surface_distances(
            self.mask1_features.region_labelled_mask,
            self.mask2_features.region_labelled_mask,
            self.mask1_features.num_objects,
            self.mask2_features.num_objects,
        )
        self.mask1_labelled_surface_distances = out[0]
        self.mask2_labelled_surface_distances = out[1]

    def compute_dice(self):
        return compute_overall_dice(
            self.mask1_features.mask, self.mask2_features.mask
        )
    
    def compute_num_detected_objects(self, detection_threshold):
        return compute_num_detected_objects(
            self.mask1_relative_overlap,
            self.mask2_relative_overlap,
            detection_threshold
        )
    
    def compute_object_accuracy(self, detection_threshold):
        return compute_object_accuracy(
            self.mask1_relative_overlap,
            self.mask2_relative_overlap,
            detection_threshold
        )

    def compute_percentie_surface_distance(self, percentile=100):
        return compute_overall_percentile_surface_distances(
            self.mask1_labelled_surface_distance,
            self.mask2_labelled_surface_distance,
            percentile,
        )

    def compute_average_surface_distances(self):
        return compute_overall_average_surface_distances(
            self.mask1_labelled_surface_distance,
            self.mask2_labelled_surface_distance,
        )


class ObjectWiseMetrics:
    def __init__(self, mask_pair_features):
        self._mask_pair_features = mask_pair_features
    
    def compute_volumes(self):
        return (
            self._mask_pair_features.mask1_features.volume,
            self._mask_pair_features.mask2_features.volume,
        )
    
    def compute_coverage_fractions(self):
        return compute_coverage_fraction(
            self._mask_pair_features.mask1_relative_overlaps,
            self._mask_pair_features.mask2_relative_overlaps,
        )

    def compute_percentile_surface_distances(self, percentile=100):
        return compute_object_percentile_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distances,
            self._mask_pair_features.mask2_labelled_surface_distances,
            percentile,
        )

    def compute_average_surface_distances(self):
        return compute_object_average_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distances,
            self._mask_pair_features.mask2_labelled_surface_distances,
        )


class OverallMetrics:
    def __init__(self, mask_pair_features, object_wise_metrics):
        self._mask_pair_features = mask_pair_features
        self._object_wise_metrics = object_wise_metrics
    
    def compute_percentile_surface_distance(self, percentile=100):
        return compute_overall_percentile_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distances,
            self._mask_pair_features.mask2_labelled_surface_distances,
            percentile,
        )

    def compute_average_surface_distance(self):
        return compute_overall_average_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distances,
            self._mask_pair_features.mask2_labelled_surface_distances,
        )
    
    def compute_dice(self):
        return compute_overall_dice(
            self._mask_pair_features.mask1_features.mask,
            self._mask_pair_features.mask2_features.mask,
        )
    
    def compute_num_overlapping_objects(self, overlap_threshold=0.5):
        return compute_num_detected_objects(
            self._mask_pair_features.mask1_relative_overlaps,
            self._mask_pair_features.mask2_relative_overlaps,
            overlap_threshold
        )
    
    def compute_object_accuracy(self, overlap_threshold=0.5):
        return compute_object_accuracy(
            self._mask_pair_features.mask1_relative_overlaps,
            self._mask_pair_features.mask2_relative_overlaps,
            overlap_threshold
        )


class MaskPairEvaluator:
    def __init__(self, mask1, mask2):
        self.mask_pair_features = MaskPairFeatures(mask1, mask2)
        self.object_wise = ObjectWiseMetrics(self.mask_pair_features)
        self.overall = OverallMetrics(self.mask_pair_features, self.object_wise)


def _compute_objectwise_reduction(
    object_wise_metrics, reduction_function, suffix, overlap_threshold, overlapping
):
    if overlap_threshold is not None and overlapping:
        cfrac = object_wise_metrics['coverage_fractions']
        overlap_indicator = cfrac > overlap_threshold
    elif overlap_threshold is not None and not overlapping:
        cfrac = object_wise_metrics['coverage_fractions']
        overlap_indicator = cfrac <= overlap_threshold
    else:
        overlap_indicator = slice(None)
    
    if suffix is None:
        suffix = ""
    else:
        suffix = f"_{suffix}"

    return {
        f"{metric_name}{suffix}": reduction_function(metric[overlap_indicator])
        for metric_name, metric in object_wise_metrics.items()
    }


def _compute_overlapping_means(object_wise_metrics, overlap_threshold):
    return _compute_objectwise_reduction(
        object_wise_metrics, np.mean, None, overlap_threshold, True
    )


def _compute_nonoverlapping_means(object_wise_metrics, overlap_threshold):
    return _compute_objectwise_reduction(
        object_wise_metrics, np.mean, None, overlap_threshold, False
    )


def compute_evaluations_for_mask_pairs(
    mask_list_1, mask_list_2, overlap_threshold=0.5, show_progress=True
):
    object_wise_metrics_1 = defaultdict(list)
    object_wise_metrics_2 = defaultdict(list)
    overall_metrics_1 = defaultdict(list)
    overall_metrics_2 = defaultdict(list)
    
    if show_progress:
        def progress(iterator):
            return tqdm(list(iterator))
    else:
        def progress(iterator):
            return iterator
    for pair_num, (mask1, mask2) in progress(enumerate(zip(mask_list_1, mask_list_2))):
        evaluator = MaskPairEvaluator(mask1, mask2)

        volumes1, volumes2 = evaluator.object_wise.compute_volumes()
        object_wise_metrics_1['volume'].extend(volumes1)
        object_wise_metrics_2['volume'].extend(volumes2)

        cfracs1, cfracs2 = evaluator.object_wise.compute_coverage_fractions()
        object_wise_metrics_1['coverage_fraction'].extend(cfracs1)
        object_wise_metrics_2['coverage_fraction'].extend(cfracs2)

        hd1, hd2 = evaluator.object_wise.compute_percentile_surface_distances(100)
        object_wise_metrics_1['hausdorff_distance'].extend(hd1)
        object_wise_metrics_2['hausdorff_distance'].extend(hd2)

        hd95_1, hd95_2 = evaluator.object_wise.compute_percentile_surface_distances(95)
        object_wise_metrics_1['95th_percentile_surface_distance'].extend(hd95_1)
        object_wise_metrics_2['95th_percentile_surface_distance'].extend(hd95_2)

        msd1, msd2 = evaluator.object_wise.compute_percentile_surface_distances(50)
        object_wise_metrics_1['median_surface_distance'].extend(msd1)
        object_wise_metrics_2['median_surface_distance'].extend(msd2)

        asd1, asd2 = evaluator.object_wise.compute_average_surface_distances()
        object_wise_metrics_1['average_surface_distance'].extend(asd1)
        object_wise_metrics_2['average_surface_distance'].extend(asd2)

        object_wise_metrics_1['mask_num'].extend([pair_num for _ in asd1])
        object_wise_metrics_2['mask_num'].extend([pair_num for _ in asd2])

        dice = evaluator.overall.compute_dice()
        overall_metrics_1['dice'].append(dice)
        overall_metrics_2['dice'].append(dice)

        hd1, hd2 = evaluator.overall.compute_percentile_surface_distance(100)
        overall_metrics_1['hausdorff_distance'].append(hd1)
        overall_metrics_2['hausdorff_distance'].append(hd2)

        hd95_1, hd95_2 = evaluator.overall.compute_percentile_surface_distance(95)
        overall_metrics_1['95th_percentile_surface_distance'].append(hd95_1)
        overall_metrics_2['95th_percentile_surface_distance'].append(hd95_2)

        msd1, msd2 = evaluator.overall.compute_percentile_surface_distance(50)
        overall_metrics_1['median_surface_distance'].append(msd1)
        overall_metrics_2['median_surface_distance'].append(msd2)

        struc_acc1, struc_acc2 = evaluator.overall.compute_object_accuracy(overlap_threshold)
        overall_metrics_1['object_accuracy'].append(struc_acc1)
        overall_metrics_2['object_accuracy'].append(struc_acc2)

    object_wise_metrics_1 = {
        metric_name: np.array(metric) for metric_name, metric in object_wise_metrics_1.items()
    }
    object_wise_metrics_2 = {
        metric_name: np.array(metric) for metric_name, metric in object_wise_metrics_2.items()
    }

    mask_1_evaluations = {
        'object_wise': pd.DataFrame(object_wise_metrics_1),
        'overall': pd.DataFrame(overall_metrics_1),
    }
    mask_2_evaluations = {
        'object_wise': pd.DataFrame(object_wise_metrics_2),
        'overall': pd.DataFrame(overall_metrics_2),
    }
    return mask_1_evaluations, mask_2_evaluations

from collections import defaultdict
from itertools import product
from numbers import Number
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from scipy.ndimage import find_objects, label
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from .overlap import (compute_coverage_fraction, compute_label_overlap,
                      compute_num_detected_objects, compute_overall_dice,
                      compute_pairwise_overlaps, compute_relative_overlaps,
                      compute_structure_accuracy, compute_structure_dice,
                      compute_volumes)
from .surface_distance import (compute_labelled_surface_distances,
                               compute_overall_average_surface_distances,
                               compute_overall_percentile_surface_distances,
                               compute_structure_average_surface_distances,
                               compute_structure_percentile_surface_distances)


class MaskFeatures:
    def __init__(self, mask):
        self.mask = mask
        self.region_labelled_mask, self.num_structures = label(mask)
        self.object_locations = find_objects(self.region_labelled_mask)
        self.volume = compute_volumes(self.region_labelled_mask, self.num_structures)


class MaskPairFeatures:
    def __init__(self, mask1, mask2):
        self.mask1_features = MaskFeatures(mask1)
        self.mask2_features = MaskFeatures(mask2)
        self.pairwise_overlap_matrix = compute_pairwise_overlaps(
            self.mask1_features.region_labelled_mask, 
            self.mask2_features.region_labelled_mask, 
            self.mask1_features.num_structures, 
            self.mask2_features.num_structures,
            self.mask1_features.object_locations,
            self.mask2_features.object_locations,
        )

        out = compute_relative_overlaps(
            self.mask1_features.volume,
            self.mask2_features.volume,
            self.pairwise_overlap_matrix,
        )
        self.mask1_relative_overlap = out[0]
        self.mask2_relative_overlap = out[1]

        # Surface distances
        out = compute_labelled_surface_distances(
            self.mask1_features.region_labelled_mask,
            self.mask2_features.region_labelled_mask,
            self.mask1_features.num_structures,
            self.mask2_features.num_structures,
        )
        self.mask1_labelled_surface_distance = out[0]
        self.mask2_labelled_surface_distance = out[1]

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
    
    def compute_structure_accuracy(self, detection_threshold):
        return compute_structure_accuracy(
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


class StructureWiseMetrics:
    def __init__(self, mask_pair_features):
        self._mask_pair_features = mask_pair_features
    
    def compute_volumes(self):
        return self._mask_pair_features.mask1_features.volume
    
    def compute_coverage_fractions(self):
        return compute_coverage_fraction(
            self._mask_pair_features.mask1_relative_overlap,
            self._mask_pair_features.mask2_relative_overlap,
        )

    def compute_percentile_surface_distances(self, percentile=100):
        return compute_structure_percentile_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distance,
            self._mask_pair_features.mask2_labelled_surface_distance,
            percentile,
        )

    def compute_average_surface_distances(self):
        return compute_structure_average_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distance,
            self._mask_pair_features.mask2_labelled_surface_distance,
        )



class OverallMetrics:
    def __init__(self, mask_pair_features, structure_wise_metrics):
        self._mask_pair_features = mask_pair_features
        self._structure_wise_metrics = structure_wise_metrics
    
    def compute_percentile_surface_distance(self, percentile=100):
        return compute_overall_percentile_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distance,
            self._mask_pair_features.mask2_labelled_surface_distance,
            percentile,
        )

    def compute_average_surface_distance(self):
        return compute_overall_average_surface_distances(
            self._mask_pair_features.mask1_labelled_surface_distance,
            self._mask_pair_features.mask2_labelled_surface_distance,
        )
    
    def compute_dice(self):
        return compute_overall_dice(
            self._mask_pair_features.mask1_features.mask,
            self._mask_pair_features.mask2_features.mask,
        )
    
    def compute_num_overlapping_objects(self, overlap_threshold=0.5):
        return compute_num_detected_objects(
            self._mask_pair_features.features_mask_1['relative_overlap'],
            self._mask_pair_features.features_mask_2['relative_overlap'],
            overlap_threshold
        )
    
    def compute_structure_accuracy(self, overlap_threshold=0.5):
        return compute_structure_accuracy(
            self._mask_pair_features.features_mask_1['relative_overlap'],
            self._mask_pair_features.features_mask_2['relative_overlap'],
            overlap_threshold
        )


class MaskPairEvaluator:
    def __init__(self, mask1, mask2):
        self.mask_pair_features = MaskPairFeatures(mask1, mask2)
        self.structure_wise = StructureWiseMetrics(self.mask_pair_features)
        self.overall = OverallMetrics(self.mask_pair_features, self.structure_wise)


def compute_overlapping_means(structure_wise_metrics, overlap_threshold):
    cfrac = structure_wise_metrics['coverage_fractions']
    overlap_indicator = cfrac > overlap_threshold

    return {
        f"{metric_name}_overlapping_mean": np.mean(metric[overlap_indicator])
        for metric_name, metric in structure_wise_metrics.items()
    }


def compute_nonoverlapping_means(structure_wise_metrics, overlap_threshold):
    cfrac = structure_wise_metrics['coverage_fractions']
    overlap_indicator = cfrac <= overlap_threshold

    return {
        f"{metric_name}_nonoverlapping_mean": np.mean(metric[overlap_indicator])
        for metric_name, metric in structure_wise_metrics.items()
    }


def compute_all_evaluations_for_mask_pairs(mask_list_1, mask_list_2, overlap_threshold=0.5):
    structure_wise_metrics_1 = defaultdict(list)
    structure_wise_metrics_2 = defaultdict(list)
    overall_metrics_1 = defaultdict(list)
    overall_metrics_2 = defaultdict(list)
    
    for mask1, mask2 in zip(mask_list_1, mask_list_2):
        evaluator = MaskPairEvaluator(mask1, mask2)

        volumes1, volumes2 = evaluator.structure_wise.compute_volumes()
        structure_wise_metrics_1['volumes'].extend(volumes1)
        structure_wise_metrics_2['volumes'].extend(volumes2)

        cfracs1, cfracs2 = evaluator.structure_wise.compute_coverage_fractions()
        structure_wise_metrics_1['coverage_fractions'].extend(cfracs1)
        structure_wise_metrics_2['coverage_fractions'].extend(cfracs2)

        hd1, hd2 = evaluator.structure_wise.compute_percentile_surface_distances(100)
        structure_wise_metrics_1['hausdorff_distance'].extend(hd1)
        structure_wise_metrics_2['hausdorff_distance'].extend(hd2)

        hd95_1, hd95_2 = evaluator.structure_wise.compute_percentile_surface_distances(95)
        structure_wise_metrics_1['95th_percentile_surface_distance'].extend(hd95_1)
        structure_wise_metrics_2['95th_percentile_surface_distance'].extend(hd95_2)

        msd1, msd2 = evaluator.structure_wise.compute_percentile_surface_distances(50)
        structure_wise_metrics_1['median_surface_distance'].extend(msd1)
        structure_wise_metrics_2['median_surface_distance'].extend(msd2)

        asd1, asd2 = evaluator.structure_wise.compute_average_surface_distances()
        structure_wise_metrics_1['average_surface_distance'].extend(asd1)
        structure_wise_metrics_2['average_surface_distance'].extend(asd2)

        dice = evaluator.overall.compute_dice()
        overall_metrics_1['dice'].append(dice)
        overall_metrics_2['dice'].append(dice)

        hd1, hd2 = evaluator.overall.compute_percentile_surface_distance(100)
        overall_metrics_1['hausdorff_distance'].extend(hd1)
        overall_metrics_2['hausdorff_distance'].extend(hd2)

        hd95_1, hd95_2 = evaluator.overall.compute_percentile_surface_distance(95)
        overall_metrics_1['95th_percentile_surface_distance'].extend(hd95_1)
        overall_metrics_2['95th_percentile_surface_distance'].extend(hd95_2)

        msd1, msd2 = evaluator.overall.compute_percentile_surface_distance(50)
        overall_metrics_1['median_surface_distance'].extend(msd1)
        overall_metrics_2['median_surface_distance'].extend(msd2)

        struc_acc1, struc_acc2 = evaluator.overall.compute_structure_accuracy(overlap_threshold)
        overall_metrics_1['structure_accuracy'].extend(struc_acc1)
        overall_metrics_2['structure_accuracy'].extend(struc_acc2)

    structure_wise_metrics_1 = {
        metric_name: np.array(metric) for metric_name, metric in structure_wise_metrics_1.items()
    }
    structure_wise_metrics_2 = {
        metric_name: np.array(metric) for metric_name, metric in structure_wise_metrics_2.items()
    }

    overlapping_structure_wise_1 = compute_overlapping_means(
        structure_wise_metrics_1, overlap_threshold
    )
    overlapping_structure_wise_2 = compute_overlapping_means(
        structure_wise_metrics_2, overlap_threshold
    )
    nonoverlapping_structure_wise_1 = compute_nonoverlapping_means(
        structure_wise_metrics_1, overlap_threshold
    )
    nonoverlapping_structure_wise_2 = compute_nonoverlapping_means(
        structure_wise_metrics_2, overlap_threshold
    )

    mask_1_evaluations = {
        'structure_wise': structure_wise_metrics_1,
        'overall': overall_metrics_1,
        'overlapping_structure_wise': overlapping_structure_wise_1,
        'nonoverlapping_structure_wise': nonoverlapping_structure_wise_1,
    }
    mask_2_evaluations = {
        'structure_wise': structure_wise_metrics_1,
        'overall': overall_metrics_1,
        'overlapping_structure_wise': overlapping_structure_wise_1,
        'nonoverlapping_structure_wise': nonoverlapping_structure_wise_1,
    }
    return mask_1_evaluations, mask_2_evaluations


def summarise_mask_metrics(
    mask_1_evaluations,
    mask_2_evaluations,
    mask_1_name,
    mask_2_name,
    overlap_threshold=0.5
):
    mask_1_summary = {
        **mask_1_evaluations['overall'],
        **mask_1_evaluations['overlapping_structure_wise'],
        **mask_1_evaluations['nonoverlapping_structure_wise'],
    }
    mask_2_summary = {
        **mask_2_evaluations['overall'],
        **mask_2_evaluations['overlapping_structure_wise'],
        **mask_2_evaluations['nonoverlapping_structure_wise'],
    }

    summary = {
        **{
            f"{mask_1_name}_{metric_name}": metric
            for metric_name, metric in mask_1_summary.items()
        },
        **{
            f"{mask_2_name}_{metric_name}": metric
            for metric_name, metric in mask_2_summary.items()
        }
    }

    return summary

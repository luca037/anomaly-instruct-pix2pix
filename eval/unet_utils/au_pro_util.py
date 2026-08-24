"""
Anomaly-localization AU-PRO metric — faithful port of O2MAG's
`eval/unet_utils/au_pro_util.py`.

This implements the *official* MVTec PRO definition (Per-Region Overlap): for
each connected component of the GT anomaly mask we compute the fraction of its
pixels that are detected, and PRO is the **mean over components** — so every
defect region counts equally regardless of its size (unlike a global pixel-TPR,
which is dominated by the largest region).

The PRO curve is integrated up to a fixed false-positive-rate limit
(`integration_limit`, default 0.3 — the standard MVTec AD protocol). Connected
components come from `scipy.ndimage.label` (scipy is installed in this repo).

Public API (matches O2MAG):
    calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100)
        -> (au_pro, pro_curve)   where pro_curve == (fprs, pros)

`calculate_aupro` is a thin backward-compatible wrapper used by
`test-localization.py`.
"""

import numpy as np
from scipy.ndimage import label
from bisect import bisect


class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing thresholds.
    """

    def __init__(self, anomaly_scores):
        # Keep a sorted list of all anomaly scores within the component.
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # Pointer to the anomaly score where the current threshold divides the
        # component into OK / NOK pixels.
        self.index = 0

        # The last evaluated threshold.
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Compute the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        # Increase the index until it points to an anomaly score that is just
        # above the specified threshold.
        while (
            self.index < len(self.anomaly_scores)
            and self.anomaly_scores[self.index] <= threshold
        ):
            self.index += 1

        # Compute the fraction of component pixels that are correctly segmented
        # as anomalous.
        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
    """
    Calculate the definite integral of a curve given by x- and y-values.
    In contrast to, e.g., 'numpy.trapz()', this function allows to define an
    upper bound to the integration range by setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a warning.
    """
    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            """WARNING: Not all x and y values passed to trapezoid are finite.
            Will continue with only the finite values."""
        )
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between the last
            # x[ins-1] and x_max. Since we do not know the exact value of y at
            # x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + (
                (y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1])
            )
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extract anomaly scores for each ground truth connected component as well as
    anomaly scores for each potential false positive pixel from anomaly maps.
    """
    # Make sure an anomaly map is present for each ground truth map.
    assert len(anomaly_maps) == len(ground_truth_maps)

    # Initialize ground truth components and scores of potential fp pixels.
    ground_truth_components = []
    n_pixels = len(ground_truth_maps) * ground_truth_maps[0].size
    anomaly_scores_ok_pixels = np.zeros(n_pixels)

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    # Collect anomaly scores within each ground truth region and for all potential
    # fp pixels.
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)

        # Store all potential fp scores.
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index : ok_index + num_ok_pixels] = prediction[
            labeled == 0
        ].copy()
        ok_index += num_ok_pixels

        # Fetch anomaly scores within each GT component.
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    # Sort all potential false positive scores.
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    """
    Compute the PRO curve at equidistant interpolation points for a set of anomaly
    maps with corresponding ground truth maps.
    """
    # Fetch sorted anomaly scores.
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(
        anomaly_maps, ground_truth_maps
    )

    # Select equidistant thresholds.
    threshold_positions = np.linspace(
        0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int
    )

    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        # Compute the false positive rate for this threshold.
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # Compute the PRO value for this threshold.
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)

    # Return (FPR/PRO) pairs in increasing FPR order.
    fprs = fprs[::-1]
    pros = pros[::-1]

    return fprs, pros


def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
    """
    Compute the area under the PRO curve for a set of ground truth images and
    corresponding anomaly images.

    Args:
        gts:         List of 2D numpy arrays with binary ground truth labels
                     (0 = anomaly-free, 1 = anomaly).
        predictions: List of 2D numpy arrays with real-valued anomaly scores.
        integration_limit: Integration limit for the area under the PRO curve.
        num_thresholds:    Number of thresholds used to sample the PRO curve.

    Returns:
        au_pro:    Area under the PRO curve computed up to the given integration
                   limit (normalized to [0, 1]).
        pro_curve: The (fprs, pros) tuple.
    """
    # Compute the PRO curve.
    pro_curve = compute_pro(
        anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds
    )

    # Compute the area under the PRO curve.
    au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit

    return au_pro, pro_curve


def calculate_aupro(gt_masks, pred_masks, fpr_points=None, num_thresholds=100):
    """Backward-compatible wrapper used by ``test-localization.py``.

    Ignores ``fpr_points`` (the PRO curve is now integrated to the O2MAG-standard
    ``integration_limit=0.3``) and returns ``(au_pro, fprs, pros)``.
    """
    au_pro, (fprs, pros) = calculate_au_pro(
        gt_masks, pred_masks, integration_limit=0.3, num_thresholds=num_thresholds
    )
    return au_pro, list(fprs), list(pros)

"""
prdc from https://github.com/clovaai/generative-evaluation-prdc
Copyright (c) 2020-present NAVER Corp.
MIT license
Modified to also report realism score from https://arxiv.org/abs/1904.06991
"""

import logging
import sys

import numpy as np
import sklearn.metrics

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

__all__ = ["compute_prdc"]


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    dists = sklearn.metrics.pairwise_distances(
        data_x,
        data_y,
        metric="euclidean",
        n_jobs=1 if len(data_x) < 10_000 else -1,
    )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_NND(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc_old(real_features, fake_features, nearest_k=None, realism=False):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int. If None, set to sqrt of number of samples used for calculation.
        realism: bool. If True, compute realism score.
    Returns:
        dict of precision, recall, density, and coverage.
    """

    n_real, n_fake = int(real_features.shape[0]), int(fake_features.shape[0])
    print(f"Num real: {n_real} Num fake: {n_fake}")

    if nearest_k is None:
        nearest_k = int(np.sqrt(n_real))
        print(f"k is None. Setting it to sqrt of num samples: {nearest_k}")
    else:
        print(f"k: {nearest_k}")

    real_NND = compute_NND(real_features, nearest_k)
    fake_NND = compute_NND(fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    P = (distance_real_fake < np.expand_dims(real_NND, axis=1)).any(axis=0).mean()
    R = (distance_real_fake < np.expand_dims(fake_NND, axis=0)).any(axis=1).mean()

    D = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_NND, axis=1)
    ).sum(axis=0).mean()
    C = (distance_real_fake.min(axis=1) < real_NND).mean()

    d = dict(
        P=P,
        R=R,
        D=D,
        C=C,
        n_real=n_real,
        n_fake=n_fake,
    )

    if realism:
        """
        Large errors, even if they are rare, would undermine the usefulness of the metric.
        We tackle this problem by discarding half of the hyperspheres with the largest radii.
        In other words, the maximum in Equation 3 is not taken over all φr ∈ Φr but only over 
        those φr whose associated hypersphere is smaller than the median.
        """
        mask = real_NND < np.median(real_NND)

        d["realism"] = (
            np.expand_dims(real_NND[mask], axis=1) / distance_real_fake[mask]
        ).max(axis=0)

    return d


def compute_prdc(
    real_feats,
    fake_feats,
    real_labs=None,
    fake_labs=None,
    nearest_k=None,
    derive_labelwise=False,
    realism=False,
):
    """
    Default nearest_k is set to sqrt(N).
    """

    if realism:
        raise NotImplementedError("Realism score not implemented.")

    if derive_labelwise and (real_labs is None or fake_labs is None):
        raise ValueError("Arg `derive_labelwise` needs `real_labs` and `fake_labs`.")

    n_real, n_fake = int(real_feats.shape[0]), int(fake_feats.shape[0])

    if nearest_k is None:
        nearest_k = int(np.sqrt(n_real))
        print(f"k is None. Setting it to sqrt of num samples: {nearest_k}")

    # Return NaN if insufficient samples
    if n_real < nearest_k + 1 or n_fake < nearest_k + 1:
        logger.warning(
            f"Insufficient samples (real: {n_real}, fake: {n_fake}, "
            f"need: {nearest_k + 1}). Returning NaN PRDC."
        )
        return dict(
            P=np.nan,
            R=np.nan,
            D=np.nan,
            C=np.nan,
            n_real=n_real,
            n_fake=n_fake,
            param_k=nearest_k,
        )

    # Compute Balls
    radii_real = compute_NND(real_feats, nearest_k)
    radii_fake = compute_NND(fake_feats, nearest_k)
    dist_real_fake = compute_pairwise_distance(real_feats, fake_feats)

    # Compute overall PRDC
    P = (dist_real_fake < np.expand_dims(radii_real, axis=1)).any(axis=0).mean()
    R = (dist_real_fake < np.expand_dims(radii_fake, axis=0)).any(axis=1).mean()
    D = (1.0 / float(nearest_k)) * (
        dist_real_fake < np.expand_dims(radii_real, axis=1)
    ).sum(axis=0).mean()
    C = (dist_real_fake.min(axis=1) < radii_real).mean()

    d = dict(
        P=P,
        R=R,
        D=D,
        C=C,
        n_real=n_real,
        n_fake=n_fake,
        param_k=nearest_k,
    )

    if derive_labelwise:
        n_labels = np.max([np.max(real_labs), np.max(fake_labs)]) + 1

        for k in range(n_labels):
            label_key = f"label-{k}"
            print(f"\n--- {label_key} (rcf) ---")

            mask_real_k = real_labs == k
            mask_fake_k = fake_labs == k

            n_real_k = np.sum(mask_real_k)
            n_fake_k = np.sum(mask_fake_k)

            # Return NaN if insufficient samples for this label
            if n_real_k < nearest_k + 1 or n_fake_k < nearest_k + 1:
                logger.warning(
                    f"{label_key}: Insufficient samples (real: {n_real_k}, fake: {n_fake_k}, "
                    f"need: {nearest_k + 1}). Returning NaN."
                )
                d[label_key] = dict(
                    P=np.nan,
                    R=np.nan,
                    D=np.nan,
                    C=np.nan,
                    n_real=n_real_k,
                    n_fake=n_fake_k,
                    param_k=nearest_k,
                )
                continue

            real_radii_k = radii_real[mask_real_k]
            fake_radii_k = radii_fake[mask_fake_k]
            dist_real_fake_k = dist_real_fake[np.ix_(mask_real_k, mask_fake_k)]

            P_k = (
                (dist_real_fake_k < np.expand_dims(real_radii_k, axis=1))
                .any(axis=0)
                .mean()
            )
            R_k = (
                (dist_real_fake_k < np.expand_dims(fake_radii_k, axis=0))
                .any(axis=1)
                .mean()
            )
            D_k = (1.0 / float(nearest_k)) * (
                dist_real_fake_k < np.expand_dims(real_radii_k, axis=1)
            ).sum(axis=0).mean()
            C_k = (dist_real_fake_k.min(axis=1) < real_radii_k).mean()

            d[label_key] = dict(
                P=P_k,
                R=R_k,
                D=D_k,
                C=C_k,
                n_real=n_real_k,
                n_fake=n_fake_k,
                param_k=nearest_k,
            )

    return d

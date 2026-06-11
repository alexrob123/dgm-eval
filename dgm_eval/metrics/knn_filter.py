import numpy as np

from .prdc import compute_NND, compute_pairwise_distance


def compute_knn_filter(rr, rg, lr, lg, nlabels, nearest_k):
    """
    Computes PRDC metrics per label using the knn-balls-filtering approach.

    Args:
        rr: numpy.ndarray([N, feature_dim], dtype=np.float32)
        rg: numpy.ndarray([N, feature_dim], dtype=np.float32)
        lr: numpy.ndarray([N], dtype=np.int32)
        lg: numpy.ndarray([N], dtype=np.int32)
        nlabels: int
        nearest_k: int
    Returns:
        dict of PRDC metrics.
    """

    n_real, n_fake = int(rr.shape[0]), int(rg.shape[0])
    print(f"Num real: {n_real} Num fake: {n_fake}")

    if nearest_k is None:
        nearest_k = int(np.sqrt(rr.shape[0]))
        print(f"k is None. Setting it to sqrt of num samples: {nearest_k}")
    else:
        print(f"k: {nearest_k}")

    # Check if we have enough per label samples
    for lab in range(nlabels):
        if np.sum(lr == lab) < nearest_k + 1:
            raise ValueError(
                f"Not enough real samples for label {lab} to compute knn balls. "
                f"Found {np.sum(lr == lab)}, but need at least {nearest_k + 1}."
            )
        if np.sum(lg == lab) < nearest_k + 1:
            raise ValueError(
                f"Not enough fake samples for label {lab} to compute knn balls. "
                f"Found {np.sum(lg == lab)}, but need at least {nearest_k + 1}."
            )

    # Compute PRDC overall
    real_radii = compute_NND(rr, nearest_k)
    fake_radii = compute_NND(rg, nearest_k)

    dist_real_fake = compute_pairwise_distance(rr, rg)

    P = (dist_real_fake < np.expand_dims(real_radii, axis=1)).any(axis=0).mean()
    R = (dist_real_fake < np.expand_dims(fake_radii, axis=0)).any(axis=1).mean()
    D = (1.0 / float(nearest_k)) * (
        dist_real_fake < np.expand_dims(real_radii, axis=1)
    ).sum(axis=0).mean()
    C = (dist_real_fake.min(axis=1) < np.expand_dims(fake_radii, axis=0)).mean()

    d = dict(
        p=P,
        r=R,
        # d=D,
        # c=C,
        nreal=n_real,
        nfake=n_fake,
    )

    # 3. Compute PRDC per label
    for i in range(nlabels):
        label = f"label-{i}"
        print(f"\n--- {label} ---")

        # filter radii for current label
        # filter rr and rg for current label
        # compute distance between filtered rr and rg
        # compare to filtered radii

        real_radii_i = real_radii[lr == i]
        fake_radii_i = fake_radii[lg == i]
        rr_i = rr[lr == i]
        rg_i = rg[lg == i]
        dist_rr_rg_i = compute_pairwise_distance(rr_i, rg_i)

        P_i = (dist_rr_rg_i < np.expand_dims(real_radii_i, axis=1)).any(axis=0).mean()
        R_i = (dist_rr_rg_i < np.expand_dims(fake_radii_i, axis=0)).any(axis=1).mean()
        # D_i = (1.0 / float(nearest_k)) * (
        #     dist_rr_rg_i < np.expand_dims(real_radii_i, axis=1)
        # ).sum(axis=0).mean()
        # C_i = (dist_rr_rg_i.min(axis=1) < np.expand_dims(fake_radii_i, axis=0)).mean()

        d_i = dict(
            p=P_i,
            r=R_i,
            # d=D_i,
            # c=C_i,
            nreal=int(rr_i.shape[0]),
            nfake=int(rg_i.shape[0]),
        )
        d[label] = d_i

    return d

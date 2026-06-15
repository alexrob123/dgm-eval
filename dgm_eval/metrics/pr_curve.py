"""
Sykes 2024
https://arxiv.org/abs/2405.01611
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

PR_CURVE_CLFS = ["cov", "ipr", "knn", "parzen"]

# --------------------------------------------------------------------------------
# Coverage
# (Naem 2020)
# https://arxiv.org/abs/2002.09797
# --------------------------------------------------------------------------------


def coverage_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """
    For each probe z:
      - compute radius r_Y(z) = dist to k-th NN of z in Y
      - compute radius r_X(z) = dist to k-th NN of z in X
      - score = #{x∈X inside B^Y_kNN(z)} / #{y∈Y inside B^X_kNN(z)}
    """

    nn_X = NearestNeighbors(n_neighbors=k).fit(X)
    nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

    dist_z_to_X, _ = nn_X.kneighbors(z)
    dist_z_to_Y, _ = nn_Y.kneighbors(z)
    r_X = dist_z_to_X[:, -1]  # dist from z to k-th neighbor in X
    r_Y = dist_z_to_Y[:, -1]  # dist from z to k-th neighbor in Y

    # distance from every x∈X to every z, and every y∈Y to every z
    dist_X_to_z, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
    dist_Y_to_z, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

    # #{x∈X : dist(x,z) <= r_Y(z)}
    # #{y∈Y : dist(y,z) <= r_X(z)}
    n_X_in_Y_ball = (dist_X_to_z <= r_Y[:, None]).sum(axis=1).astype(float)
    n_Y_in_X_ball = (dist_Y_to_z <= r_X[:, None]).sum(axis=1).astype(float)

    return n_X_in_Y_ball / (n_Y_in_X_ball + 1e-12)


# --------------------------------------------------------------------------------
# Improved PR (adapative bandwidth Kernel Density Estimator)
# (Kynkaanniemi 2019)
# https://arxiv.org/abs/1904.06991
# --------------------------------------------------------------------------------


# def ipr_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
#     """
#     For each probe point z, compute p̂(z) and q̂(z) via manifold indicator,
#     then return the log-ratio score (monotone in p̂/q̂, avoids division by zero).
#     """

#     nn_X = NearestNeighbors(n_neighbors=k).fit(X)
#     nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

#     distances_X, _ = nn_X.kneighbors(X)
#     distances_Y, _ = nn_Y.kneighbors(Y)
#     knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
#     knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

#     # distances from each z to every x / y
#     dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
#     dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

#     p_hat = (dist_z_to_X <= knn_radii_X[None, :]).sum(axis=1).astype(float)
#     q_hat = (dist_z_to_Y <= knn_radii_Y[None, :]).sum(axis=1).astype(float)

#     return p_hat / (q_hat + 1e-12) # (|z|,)


def ipr_score(
    z: np.ndarray,  # (|z|, d)
    X: np.ndarray,  # (|X|, d)
    Y: np.ndarray,  # (|Y|, d)
    k: int,
    n_jobs: int = 8,
) -> np.ndarray:
    """
    For each probe point z, compute p̂(z) and q̂(z) via manifold indicator,
    then return the score (monotone in p̂/q̂, avoids division by zero).
    """

    nn_X = NearestNeighbors(
        n_neighbors=k + 1,  # do not count itself
        algorithm="ball_tree",  # tree indexing
        n_jobs=n_jobs,
    ).fit(X)
    nn_Y = NearestNeighbors(
        n_neighbors=k + 1,  # do not count itself
        algorithm="ball_tree",  # tree indexing
        n_jobs=n_jobs,
    ).fit(Y)

    distances_X, _ = nn_X.kneighbors(X)
    distances_Y, _ = nn_Y.kneighbors(Y)
    knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X, (|X|,)
    knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y, (|Y|,)

    # distances from each z to every x / y
    # dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
    # dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

    # Replace kneighbors(n_neighbors=len(X)) with radius_neighbors for efficiency
    # use global max radius for parallelism, then filter per point
    dists_z_X, idx_z_X = nn_X.radius_neighbors(
        z,
        radius=knn_radii_X.max(),
        return_distance=True,
    )
    dists_z_Y, idx_z_Y = nn_Y.radius_neighbors(
        z,
        radius=knn_radii_Y.max(),
        return_distance=True,
    )

    # Count matches via per-point radius filtering
    n_z = len(z)
    p_hat = np.zeros(n_z, dtype=np.float64)  # (|z|,)
    q_hat = np.zeros(n_z, dtype=np.float64)  # (|z|,)

    for i in range(n_z):
        # keep only x's where dist <= that x's own radius (the manifold indicator)
        p_hat[i] = (dists_z_X[i] <= knn_radii_X[idx_z_X[i]]).sum()
        q_hat[i] = (dists_z_Y[i] <= knn_radii_Y[idx_z_Y[i]]).sum()

    return p_hat / (q_hat + 1e-12)  # (|z|,)


# --------------------------------------------------------------------------------
# KNN (KNN Classifier)
# (Park & Kim 2023)
# https://arxiv.org/abs/2309.01590
# --------------------------------------------------------------------------------


def knn_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """
    For each probe z, find k nearest neighbours in X∪Y.
    Score = (# that come from X) - (# that come from Y).
    High score → neighbourhood is mostly real → z is real-like.
    """
    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

    # for each z, count how many of its k nearest neighbours are real vs fake
    nn = NearestNeighbors(n_neighbors=k).fit(XY)
    _, indices = nn.kneighbors(z)  # (|z|, k)

    neighbour_labels = labels[indices]
    n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
    n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

    return n_X / (n_Y + 1e-12)


# --------------------------------------------------------------------------------
# Parzen (fixed bandwidth Kernel Density Estimator)
# --------------------------------------------------------------------------------


def parzen_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Fixed-bandwidth Parzen classifier.
    Same as IPR but with a single global radius per dataset
    instead of per-sample adaptive radii.
    ρ_X = mean kNN radius over X
    ρ_Y = mean kNN radius over Y
    """

    nn_X = NearestNeighbors(n_neighbors=k).fit(X)
    nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

    distances_X, _ = nn_X.kneighbors(X)
    distances_Y, _ = nn_Y.kneighbors(Y)
    knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
    knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

    rho_X = np.mean(knn_radii_X)  # single global bandwidth for P
    rho_Y = np.mean(knn_radii_Y)  # single global bandwidth for Q

    # distances from each z to every x / y
    dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
    dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

    p_hat = (dist_z_to_X <= rho_X).sum(axis=1).astype(float)  # (|z|,)
    q_hat = (dist_z_to_Y <= rho_Y).sum(axis=1).astype(float)  # (|z|,)

    return p_hat / (q_hat + 1e-12)


# --------------------------------------------------------------------------------
# PR Curve
# --------------------------------------------------------------------------------


SCORE = {
    "cov": coverage_score,
    "ipr": ipr_score,
    "knn": knn_score,
    "parzen": parzen_score,
}


def compute_pr_curve(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: np.ndarray,
    clf: str,
    k: int,
):
    P_scores = SCORE[clf](X, X, Y, k)  # (N,)
    Q_scores = SCORE[clf](Y, X, Y, k)  # (N,)

    thresholds = 1.0 / lambdas  # (L,)
    fpr = (P_scores[:, None] < thresholds[None, :]).mean(axis=0)  # (L,)
    fnr = (Q_scores[:, None] >= thresholds[None, :]).mean(axis=0)  # (L,)

    risk = lambdas[:, None] * fpr[None, :] + fnr[None, :]  # (L, L)
    precisions = risk.min(axis=1)  # (L,)
    recalls = precisions / lambdas  # (L,)

    return precisions, recalls

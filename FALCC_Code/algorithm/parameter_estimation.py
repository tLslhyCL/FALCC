"""Implements the LOG-Means [1] algorithm for parameter estimation of KMeans.
[1] Fritz, M., Behringer, M., Schwarz, H. "LOG-Means: Efficiently Estimating
    the Number of Clusters in Large Datasets". 2020.
"""
from sklearn.cluster import KMeans


def log_means(X, k_low, k_high):
    """Uses the LOG-means algorithm to estimate the perfect amount of clusters.

    Parameters
    ----------
    X: DataFrame, shape (n_samples, m_features)
        Dataset on which clustering will be performed.

    k_low: int
        The minimum amount of clusters that should be generated.

    k_high: int
        The maximum amount of clusters that should be generated.


    Returns/Output
    ----------
    parameter_best: int
        Returns the best estimated parameter within the given range [k_low, k_high].
    """
    k_low = k_low - 1
    K = []
    M = {}
    SSE_low = KMeans(k_low).fit(X).inertia_
    K.append((k_low, SSE_low))
    SSE_high = KMeans(k_high).fit(X).inertia_
    K.append((k_high, SSE_high))
    count = 0
    while k_high != k_low + 1:
        k_mid = int((k_high + k_low)/2)
        SSE_mid = KMeans(k_mid).fit(X).inertia_
        K.append((k_mid, SSE_mid))
        ratio_left = SSE_low/SSE_mid
        ratio_right = SSE_mid/SSE_high
        if ratio_left >= ratio_right:
            k_high = k_mid
            #k_low = k_low
        else:
            #k_high = k_high
            k_low = k_mid
        K = sorted(K, key=lambda x: x[0])
        for pos, t in enumerate(K):
            if t[0] == k_high:
                k_high_id = pos
                break
        SSE_high = K[k_high_id][1]
        SSE_low = K[k_high_id-1][1]
        count += 1

    try:
        if float(M[k_high]) >= float(M[k_low]):
            parameter_best = k_high
        else:
            parameter_best = k_low
    except KeyError:
        parameter_best = k_high

    return parameter_best

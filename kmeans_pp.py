"""
    kmeans_pp.py chooses the initial centroids for the K-means algorithm and returns the clusters made by it.
"""

import numpy as np
import mykmeanssp as km

"""
calculate the initial centroids and the clusters made by the Kmeans algorithm
params: K- the number of clusters
        N- the number of observations
        d- the dimendion of the observations
        observationArr- the observations
return: clusters - the clusters made by the Kmeans algorithm
"""

def k_means_pp(K, N, d, observationsArr):
    np.random.seed(0)
    centroids = np.zeros((K, d))
    ind = np.random.choice(N)
    centroids[0] = observationsArr[ind]
    distances = np.zeros((K, N))
    distances[0] = np.power((observationsArr - centroids[0]), 2).sum(axis=1)
    for i in range(1, K):
        probs = np.min(distances[:i,], axis=0)
        probsSum = probs / probs.sum()
        ind = np.random.choice(N, p = probsSum)
        centroids[i] = observationsArr[ind]
        distances[i] = np.power((observationsArr - centroids[i]), 2).sum(axis=1)
    observationsArr = observationsArr.tolist()
    centroids = centroids.tolist()
    clusters = km.kmeanspp(observationsArr, centroids, K, N, d)
    return clusters
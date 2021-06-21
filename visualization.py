"""
    visualization.py calculates the Jaccard Measure for the clusters made by the K-means Algorithm and the
    Normalized Spectral Clustering Algorithm. Also, it creates the visualization of the results.
"""

import numpy as np
import matplotlib.pyplot as plt

"""
    calculate the Jaccard Measure
    params: arr1- the clusters made by the Kmeans++ Algorithm
            arr2- the clusters made by make_blobs
    return:the Jaccard Measure for arr1, arr2
"""


def jaccard(arr1, arr2):
    nom = 0
    dem = 0
    for i in range (len(arr1)):
        for j in range (i+1, len(arr1)):
            if (arr1[i] == arr1[j] and arr2[i] == arr2[j]):
                nom += 1
            if (arr1[i] == arr1[j] or arr2[i] == arr2[j]):
                dem += 1
    return nom / dem


def createString(n, k, K, jac_ns, jac_k):
    s = "Data was generated from the values:\n" \
        "n = {}, k = {}\n" \
        "The k that was used for both algorithms was {}\n" \
        "The Jaccard measure for Spectral Clustering: {}\n" \
        "The Jaccard measure for K-means: {}".format(n,k,K,jac_ns,jac_k)
    return s

"""
    create the visualization of the algorithm's results
    params: obs- nparray with the samples made by make_blobs
            nsClustering- the clusters made by the Kmeans++ Algorithm and the Eigengap Heuristic
            kMeansClustering- the clusters made by the Kmeans++ Algorithm and the observations made by make_blobs
            labels- the clusters made by make_blobs
"""

def graphing(obs, nsClustering, kMeansClustering, labels):
    fig = plt.figure()
    if (len(obs[0]) == 2):
        plt.subplot(121)
        plt.scatter(obs[:, 0], obs[:, 1], c=nsClustering)
        plt.title("Normalized Spectral Clustering")
        plt.subplot(122)
        plt.scatter(obs[:, 0], obs[:, 1], c=kMeansClustering)
        plt.title("K-means")
    else:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title("Normalized Spectral Clustering")
        ax1.scatter(obs[:, 0], obs[:, 1],obs[:, 2], s=20, c=nsClustering)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("K-means")
        ax2.scatter(obs[:, 0], obs[:, 1],obs[:, 2], s=20, c=kMeansClustering)
    jac_ns = jaccard(nsClustering,labels)
    jac_k = jaccard(kMeansClustering, labels)
    if len(obs[0]) == 2:
        plt.figtext(0.5, -0.2, s=createString(len(obs), np.max(labels)+1, np.max(nsClustering)+1, "{:.2f}".format(jac_ns), "{:.2f}".format(jac_k)), fontsize=13, ha='center')
    else:
        plt.figtext(0.54, -0.1, s=createString(len(obs), np.max(labels)+1, np.max(nsClustering)+1, "{:.2f}".format(jac_ns), "{:.2f}".format(jac_k)), fontsize=13, ha='center')
    fig.savefig("clusters.pdf", bbox_inches='tight')



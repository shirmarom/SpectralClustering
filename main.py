"""
    main.py implements the K-means Algorithm and the Normalized Spectral Clustering Algorithm.
    It creates the T matrix and the k found by the Eigengap Heuristic and visualizes the results.
    global params: REL_K: the value of k used by the K-means Algorithm and Normalized Spectral Clustering Algorithm.
                   MAXCAP_N2, MAXCAP_K2: the max capacity for the case d=2
                   MAXCAP_N3, MAXCAP_K3: the max capacity for the case d=3
                   We have maximized n (the number of observations). We determined k = 20 because it didn't affect the
                   running time that much, and for k= 2 0 we have received better Jaccard Measures.
"""

import NormalizedClusteringFunctions as nsc
from sklearn.datasets import make_blobs
from random import randrange
import argparse
import kmeans_pp as kmpp
import numpy as np
import visualization as vis


REL_K = 0
MAXCAP_N2, MAXCAP_K2 = 540, 20
MAXCAP_N3, MAXCAP_K3 = 530, 20

print("The maximum capacity of this project for a 2-dimensional data is: K - ", MAXCAP_K2, " N - ", MAXCAP_N2, "\n")
print("The maximum capacity of this project for a 3-dimensional data is: K - ", MAXCAP_K3, " N - ", MAXCAP_N3)

"""
   determine the matrix T and the value of k
   params: X- nparray with the samples made by make_blobs
           REL_K- the temporary value of k
   return: T- the matrix T
           REL_k- the value of k used by the Kmeans++ Algorithm
"""

def normalizedSpectralClustering(obs, REL_K):
    W = nsc.adjacency_matrix(obs)
    D = nsc.diagonal_degree(W)
    L = nsc.laplacian(W, D)
    k, U = nsc.eigengap(L)
    if args.Random == "True":
        REL_K = k
    T = nsc.normalizingU(U[:, :REL_K])
    return T, REL_K


"""
    running the Kmeans++ Algorithm, and creating the visualization and the output files
    params: X- nparray with the samples made by make_blobs
            REL_K- the temporary value of k
"""

def create_files(obs, REL_K):
    T, updatedK = normalizedSpectralClustering(obs, REL_K)
    try:
        nscClusters = np.array(kmpp.k_means_pp(updatedK, n, updatedK, T))
    except:
        print("There was an error with the K-means algorithm!")
        exit(0)
    try:
        kmeansClusters = np.array(kmpp.k_means_pp(updatedK, n, d, obs))
    except:
        print("There was an error with the K-means algorithm!")
        exit(0)
    create_data_file(labels)
    create_clusters_file(nscClusters, kmeansClusters, updatedK)
    vis.graphing(obs, nscClusters, kmeansClusters, labels)


"""
    creating the data.txt file
    params: clusters- the clusters made by make_blobs
"""


def create_data_file(clusters):
    numpyClusters = np.array(clusters).reshape(n, 1)
    mat = np.concatenate((obs, numpyClusters), axis=1)
    np.savetxt('data.txt', mat, fmt=",".join(['%s'] * d + ['%i']))


"""
    creating the clusters.txt file
    params: clusters- the clusters made by the Kmeans++ Algorithm
"""


def create_clusters_file(nsc, kmeans, updatedK):
    np_clusters = np.array(nsc)
    kmeans_clusters = np.array(kmeans)
    with open('clusters.txt', "w") as outfile:
        outfile.write(str(updatedK) + '\n')
        for i in range(0, updatedK):
            outfile.write(str((np.where(np_clusters == i)[0].tolist()))[1:-1])
            outfile.write('\n')
        for i in range(0, updatedK):
            outfile.write(str((np.where(kmeans_clusters == i)[0].tolist()))[1:-1])
            if i < updatedK - 1:
                outfile.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('K', type=int)
    parser.add_argument('N', type=int)
    parser.add_argument('Random', type=str)
    args = parser.parse_args()
    d = randrange(2, 4)

    if args.Random not in ["True", "False"]:
        print("The arguments you have given are invalid. Please make sure that random is either true or false.")
        exit(0)

    if args.Random == "True":
        if d == 2:
            n = randrange(MAXCAP_N2 / 2, MAXCAP_N2)
            K = randrange(MAXCAP_K2 / 2, MAXCAP_K2)
        else:
            n = randrange(MAXCAP_N3 / 2, MAXCAP_N3)
            K = randrange(MAXCAP_K3 / 2, MAXCAP_K3)
        obs, labels = make_blobs(n_samples=n, n_features=d, centers=K)
    else:
        if args.K < 1 or args.N < 1 or args.K >= args.N:
            print("The arguments you have given are invalid. Please make sure that: " + '\n')
            print("1. k <= n." + '\n')
            print("2. both k and n are at least 1, and both of them are natural numbers only." + '\n')
            exit(0)
        n = args.N
        K = args.K
        REL_K = K
        obs, labels = make_blobs(n_samples=n, n_features=d, centers=K)
    create_files(obs, REL_K)




"""
    NormalizedClusteringFunctions.py implements the functions used by the Normalized Spectral Clustering Algorithm.
"""

import numpy as np

epsilon = 0.0001

"""
    create the Weighted Adjacency Matrix
    params: obs- nparray with the samples made by make_blobs
    return: W- the weighted adjacency matrix
"""


def adjacency_matrix(obs):
    n = len(obs)
    W = np.zeros(shape=(n,n))
    for i in range(n):
        W[i] = np.exp(np.linalg.norm(obs-obs[i],axis=1) / (-2))
    np.fill_diagonal(W, 0)
    return W

"""
    create the Diagonal Degree Matrix
    params: mat- the weighted adjacency matrix as nparray
    return: D- the diagonal degree matrix
"""

def diagonal_degree(mat):
    n = len(mat)
    D = np.zeros(shape=(n,n))
    diag = np.sum(mat, axis=1)
    np.fill_diagonal(D, diag)
    return D

"""
    raise the matrix to the (-1/2) power
    params: mat- the diagonal degree matrix as nparray
    return: D ^ (-1/2)
"""


def minus_square_mat(mat):
    res = np.diag(np.power(np.diag(mat), -0.5))
    return res

"""
    create the Normalized Graph Laplacian
    params: W- the weighted adjacency matrix as nparray
            D- the diagonal degree matrix as nparray
    return: I - (D^(-1/2) @ W @ D^(-1/2))
"""


def laplacian(W, D):
    d_square = minus_square_mat(D)
    identity_mat = np.identity(len(W))
    return identity_mat - np.dot(np.dot(d_square, W), d_square)

"""
    The Modified Gram-Schmidt Algorithm
    params: A- as nparray
    return: Q - the orthogonal matrix
            R - the upper triangular matrix
    If the norm = 0, Q[:, i] remains vector of zeros and we continue running
"""

def gram_schmidt(A):
    U = A.copy()
    n = len(A)
    R = np.zeros(shape=(n,n))
    Q = np.zeros(shape=(n,n))
    for i in range (n):
        R[i][i] = np.linalg.norm(U[:, i], 2)
        if R[i][i] != 0:
            Q[:, i] = U[:, i] / R[i][i]
        Qi = Q[:, i]
        R[i, i + 1:n] = np.einsum('i,ij->j', Qi, U[:, i + 1:n])
        U[:, i + 1:n] -= np.einsum('i,j->ji', R[i, i + 1:n], Qi)
    return Q, R


"""
    The QR Iteration Algorithm
    params: a- matrix as nparray
    return: a_hat- whose diagonal elements approach the eigenvalues of a.
            q_hat- whose columns approach the eigenvectors of a.
"""


def qr_iteration(A):
    n = len(A)
    A_hat = np.copy(A)
    Q_hat = np.identity(n)
    for i in range(n):
        q, r = gram_schmidt(A_hat)
        A_hat = np.dot(r,q)
        q_hat_q = np.dot(Q_hat, q)
        if (np.abs(np.subtract(np.abs(Q_hat), np.abs(q_hat_q))) <= epsilon).all():
            return A_hat, Q_hat
        Q_hat = q_hat_q
    return A_hat, Q_hat


"""
    calculate k and U
    params: l_norm- the Normalized Graph Laplacian
    return: k- the value of k that was found by the Eigengap Heuristic
            eigenVectors- the matrix U whose columns approach the eigenvectors of l_norm
"""


def eigengap(l_norm):
    a_hat, q_hat = qr_iteration(l_norm)
    diagonal = a_hat.diagonal().copy()
    vectorValuesSorted = np.argsort(diagonal)
    diagonal.sort()
    eigenVectors = q_hat[:, vectorValuesSorted]
    eigengaps = np.abs(diagonal[:-1] - diagonal[1:])
    relevant_gaps = eigengaps[:int(np.ceil(len(diagonal) / 2))]
    return np.argmax(relevant_gaps) + 1, eigenVectors


"""
    renormalizing the matrix U
    params: mat- the matrix U
    return: T- the matrix U after renormalizing it
"""


def normalizingU(U):
    rowsSum = np.power(np.power(U, 2).sum(axis=1), 0.5)
    return np.divide(U, rowsSum[:, None])



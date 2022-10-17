import networkx as nx
import numpy as np
import scipy.sparse as sparse
import theano
import torch
from scipy.sparse import csgraph
from theano import tensor as T


def direct_compute_deepwalk_matrix(A, window, b=1):
    '''Calculate deepwalk matrix for give adjacency matrix
    https://github.com/xptree/NetMF/blob/master/netmf.py
    Parameters
    ----------
    A : scipy.sparse.csr.csr_matrix
        adjacency matrix
    window : int
        window size
    b : int
        the number of negative samplin
    '''
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    X = sparse.identity(n) - L  # D^(-1/2) @ A @ D^(-1/2)
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
        https://github.com/ylaboratory/ETNA/blob/master/src/algorithms/helper.py
    """

    # Laplacian
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)

    A = nx.adjacency_matrix(g)
    N = sparse.diags(np.array(g.degree())[:,1].clip(1) ** -0.5, dtype=float)
    L = sparse.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sparse.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sparse.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return lap_pos_enc


def edgeindex2adjacency(mask1, mask2, edge_index, edge_index12, total_num):

    target_matrix = np.zeros([total_num, total_num])

    for e in edge_index:
        target_matrix[e[0], e[1]] = 1
        target_matrix[e[1], e[0]] = 1

    return target_matrix[mask1,:][:,mask2]

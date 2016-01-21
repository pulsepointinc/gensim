#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
from numpy import empty as empty_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from six.moves import xrange

try:
    from numpy import VisibleDeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)
except ImportError:
    pass

from numpy import *

def powerMethodBase(A,x0,iter):
  """ basic power method """
  for i in range(iter):
    x0 = dot(A,x0)
    x0 = x0/linalg.norm(x0,1)
  return x0


def maximalEigenvector(A):
  """ using the eig function to compute eigenvectors """
  n = A.shape[1]
  w,v = linalg.eig(A)
  # print "NP.EIG:", w
  ww = [(i,w[i]) for i in xrange(len(w))]
  ww.sort(key=lambda x: linalg.norm(x[1]), reverse=True)
  # print ww
  i = ww[0][0] 
  return abs(real(v[:n,i])/linalg.norm(v[:n,i],1))


def getTeleMatrix(A,m):
  """ return the matrix M
      of the web described by A """
  n = A.shape[1]
  S = ones((n,n))/n
  return (1-m)*A+m*S


def pagerank_weighted_scipy(graph, damping=0.85):
    adjacency_matrix = build_adjacency_matrix(graph)
    probability_matrix = build_probability_matrix(graph)

    pagerank_matrix = damping * adjacency_matrix.todense() + (1 - damping) * probability_matrix
    vals, vecs = eig(pagerank_matrix, left=True, right=False)
    # print vals
    # print vecs
    return process_results_scipy(graph, vecs)



def pagerank_weighted(graph, damping=0.85):
    if len(graph.nodes()) == 0: #empty graph
       return process_results(graph, [])
    adjacency_matrix = build_adjacency_matrix(graph)
    probability_matrix = build_probability_matrix(graph)

    pagerank_matrix = damping * adjacency_matrix.todense() + (1 - damping) * probability_matrix
    # print (pagerank_matrix - pagerank_matrix.T).sum(), pagerank_matrix.shape, (pagerank_matrix == pagerank_matrix.T).all()
    vals, vecs = eigs(pagerank_matrix.T, k=1)  # TODO raise an error if matrix has complex eigenvectors?
    # print "SP.EIGS:", vals[:5]
    # print type(pagerank_matrix), pagerank_matrix.sum()
    # print pagerank_matrix
    A = array(pagerank_matrix)
    # print (A == pagerank_matrix).all()
    # print repr(A)
    my_vecs = powerMethodBase(A, [1]*pagerank_matrix.shape[0],130)
    # x3 = maximalEigenvector(A)
    # print repr(my_vecs)
    # print repr(x3)
    # print repr(vecs)
    # print "="*10
    # print repr(dot(A,my_vecs))
    # print repr(dot(A,x3))
    # print repr(dot(pagerank_matrix, vecs))
    
    
    A = array([ [0,     0,     0,     1, 0, 1],
            [1/2.0, 0,     0,     0, 0, 0],
            [0,     1/2.0, 0,     0, 0, 0],
            [0,     1/2.0, 1/3.0, 0, 0, 0],
            [0,     0,     1/3.0, 0, 0, 0],
            [1/2.0, 0,     1/3.0, 0, 1, 0 ] ])

    # n = A.shape[1] # A is n x n
    # m = 0.15
    # M = getTeleMatrix(A,m)
    
    # vals2, vecs2 = eigs(M, k=1)  # TODO raise an error if matrix has complex eigenvectors?
    # print "SP:EIGS", vals2
    # print type(M), M.sum()
    # x2 = powerMethodBase(M, [1]*n,130)
    # x3 = maximalEigenvector(M)
    # print repr(x2)
    # print repr(x3)
    # print repr(vecs2)
    # print "=========="
    # print dot(M,x2)
    # print dot(M,x3)
    # print dot(M, vecs2)
    
    return process_results(graph, my_vecs) #vecs.real)


def build_adjacency_matrix(graph):
    row = []
    col = []
    data = []
    nodes = graph.nodes()
    length = len(nodes)

    for i in xrange(length):
        current_node = nodes[i]
        neighbors_sum = sum(graph.edge_weight((current_node, neighbor)) for neighbor in graph.neighbors(current_node))
        for j in xrange(length):
            edge_weight = float(graph.edge_weight((current_node, nodes[j])))
            if i != j and edge_weight != 0.0:
                row.append(i)
                col.append(j)
                data.append(edge_weight / neighbors_sum)

    return csr_matrix((data, (row, col)), shape=(length, length))


def build_probability_matrix(graph):
    dimension = len(graph.nodes())
    matrix = empty_matrix((dimension, dimension))

    probability = 1.0 / float(dimension)
    matrix.fill(probability)

    return matrix


def process_results(graph, vecs):
    scores = {}
    for i, node in enumerate(graph.nodes()):
        scores[node] = abs(vecs[i]) # , :])
        # print node, scores[node]

    return scores

def process_results_scipy(graph, vecs):
    scores = {}
    for i, node in enumerate(graph.nodes()):
        scores[node] = abs(vecs[i][0])

    return scores

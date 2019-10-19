import networkx as nx
import numpy as np
import warnings

def cloud_to_matrix( cloud ):
    norm = np.array([ np.linalg.norm(cloud, axis=1) ])
    return np.tensordot(cloud, cloud, (1, 1)) / norm / norm.T

def compute_proxs( indices, matrix, exp, scale, floor ):
    scale = np.cos(scale)
    matrix = np.clip(matrix[indices] - scale, 0, None) / (1 - scale)
    return matrix ** exp + floor

def compute_prefs( indices, degrees, exp ):
    if np.sum(degrees) == 0:
        matrix = np.zeros((len(indices))) + 1
    else:
        matrix = degrees[indices[0]] * degrees[indices[1]]
    return (matrix / np.max(matrix)) ** exp

def compute_intgs( indices, degrees, exp, boost ):
    numerator = np.maximum( degrees[indices[0]], degrees[indices[1]] ) - 0.5
    denominator = np.minimum( degrees[indices[0]], degrees[indices[1]] ) - 1.0
    boost = (1.0 / boost) ** (1.0 / exp)
    matrix = numerator / (denominator + boost)
    return (matrix / np.max(matrix)) ** exp

def connect_edges(matrix, indices, batch, dmax):
    # update the adjacency matrix
    matrix[indices[0][batch], indices[1][batch]] = 1
    matrix[indices[1][batch], indices[0][batch]] = 1
    # cut edges where at least one node has excessive degree
    degrees = np.sum(matrix, 1)
    excess = np.where(degrees > dmax)
    matrix[excess, :] = 0
    matrix[:, excess] = 0
    return (excess[0].size, np.sum(degrees[excess]))

def weighted_link(
    indices, prox_matrix, adj_matrix,
    edges, batch_size, degree_max,
    pref_exp, intg_exp, boost
):
    reflow = 0
    while (edges > 0):
        # compute weights
        degrees = np.sum(adj_matrix, 1) + 1
        pref_matrix = compute_prefs( indices, degrees, pref_exp )
        intg_matrix = compute_intgs( indices, degrees, intg_exp, boost )
        weights = prox_matrix * pref_matrix * intg_matrix
        weights = weights / weights.sum()
        # choose a batch of pairs at weighted random and connect with edges
        if np.max(weights) == 0:
            warnings.warn("Terminating early--no valid edges can be formed.")
            break
        batch_size = int(min(batch_size, edges))
        batch = np.random.choice(indices[0].size, batch_size, p=weights, replace=False)
        (e_nodes, e_edges) = connect_edges(adj_matrix, indices, batch, degree_max)
        # book-keeping
        prox_matrix[batch] = 0
        edges -= batch.size - e_edges
        reflow += e_nodes
        if reflow > len(degrees) / 5:
            warnings.warn("Terminating early--20% of nodes have reached degree_max and been reset.")
            break
    return adj_matrix

def gen_topology(
    nodes, edges,
    dimension, pref_exp, intg_exp, loner_boost,
    prox_exp, prox_scale, prox_floor,
    cloud = None, batch_count = 50, degree_max = None
):
    if degree_max == None: degree_max = int(nodes / 2)
    if cloud == None: cloud = np.random.normal(0, 1, (nodes, dimension))
    batch_size = int(edges / batch_count)
    indices = np.triu_indices(nodes, k = 1) # select unique, non-self pairs
    dimension = np.clip(int(dimension), 2, 12)
    prox_matrix = cloud_to_matrix(cloud[:,:dimension])
    prox_matrix = compute_proxs(indices, prox_matrix, prox_exp, prox_scale, prox_floor)
    adj_matrix = np.zeros((nodes, nodes))
    G = nx.from_numpy_matrix(weighted_link(
        indices, prox_matrix, adj_matrix,
        edges, batch_size, degree_max,
        pref_exp, intg_exp, loner_boost
    ))
    return nx.Graph(max(nx.connected_component_subgraphs(G), key=len))
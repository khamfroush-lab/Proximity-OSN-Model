import numpy as np
import networkx as nx
%tensorflow_version 2.x
import tensorflow as tf
import tensorflow_probability as tfp

# zero the diagonal of a matrix:
@tf.function
def rejdiag(matrix):
    return matrix - tf.linalg.band_part(matrix, 0, 0)

# zero all but upper triangle of a matrix:
@tf.function
def utrio(matrix):
    return rejdiag(tf.linalg.band_part(matrix, 0, -1))

# generate a proximity weight matrix:
@tf.function
def gen_cloud( nodes, dimensions, offs, rho ):
    # make a gaussian point cloud
    offsv = tf.where(tf.range(dimensions) == 0, offs, 0.)
    cloud = tf.random.normal( shape = (nodes, dimensions), dtype=tf.float32 ) + offsv
    # norm it, so it is shell-like
    norm = tf.einsum('xi,xi->x', cloud, cloud) ** .5
    norm = tf.reshape(tf.repeat(norm, dimensions), (nodes, dimensions))
    cloud = cloud / norm
    # dot product between all pairs of points computes cosine distance
    #  rho controls the maximum angular distance between
    #  nodes that results in non-zero proximity
    cloud = tf.einsum('xj,yj->xy', cloud, cloud) - rho
    # metric transform
    #  zero self-proximity
    #  zero negative proximity
    #  norm for rho
    #  cube to get an s-curve
    return (tf.nn.relu(rejdiag(cloud)) / (1. - rho)) ** 3.

# generate the triangulation/path weight matrix:
@tf.function
def compute_triangs( x ):
    # using the simple exponential kernel:
    return tf.linalg.expm(x)

# generate the norm/control weight matrix:
@tf.function
def compute_norm( x, n, k ):
    # compute the degree vector
    deg = tf.einsum('xi->x', tf.cast(x, dtype=tf.float32)) + 1.
    # nodes with zero degree get high weight
    # nodes with high degree get low weight
    norm = (deg ** n) * (2. - k) / 4 / (deg - k)
    # cross-multiply to make a weight matrix
    norm = tf.einsum('x,y->xy', norm, tf.ones(norm.shape))
    return  norm * tf.transpose(norm)

# generate the weight matrix:
@tf.function
def compute_weights(adj_matrix, prox_matrix, n, k, t0, p0):
    norm = compute_norm(adj_matrix, n, k)
    path = t0 + compute_triangs(adj_matrix)
    prox = p0 + prox_matrix
    weights = norm*tf.math.log(1 + prox*path)
    # zero the lower triangle (duplicates) and pre-existing edges
    weights = utrio(tf.where(adj_matrix == False, weights, 0.))
    return weights

# generate the weight matrix:
@tf.function
def compute_final_weights(adj_matrix, prox_matrix):
    norm = compute_norm(adj_matrix, 2., .5)
    path = .2 + compute_triangs(adj_matrix)
    prox = .2 + prox_matrix
    rand = 1.5 ** tf.random.normal( shape = tf.shape(adj_matrix), dtype=tf.float32 )
    return rand * tf.where(adj_matrix == True, prox*path, 0.) / norm

# comb sampling from a weight matrix:
#  (hold on to your hats)
@tf.function
def choose(choices, n):
    s = choices.shape
    # flatten and shuffle weight matrix
    choices = tf.reshape(choices, [-1])
    shuffle = tf.random.shuffle(tf.range(s[0]*s[1]))
    choices = tf.gather(choices, shuffle)
    # convert to cdf
    choices = tf.math.cumsum(choices)
    choices /= choices[-1]
    # build a comb of spaced samples
    samples = tf.random.uniform(
        minval = 0., maxval = 1./n,
        shape = (), dtype=tf.float32 )
    samples = tf.range(start=samples, limit=1., delta=1./n)
    # find them in the cdf
    choices = tf.searchsorted(choices, samples, side='right')
    choices = tf.math.minimum(choices, s[0]*s[1]-1)
    choices = tf.unique(choices)[0]
    # unshuffle the indices
    # choices = tf.gather(shuffle, choices)
    choices = (choices // s[0], choices % s[0])
    return tf.transpose( tf.stack(choices) )

# turn a list of indices into a one-hot matrix:
@tf.function
def to_update(x, choices, nchoices, nnodes):
    choices = tf.scatter_nd( choices, tf.ones(nchoices, dtype=tf.bool), (nnodes,nnodes) )
    # add transitive edges
    return tf.math.logical_or(choices, tf.transpose(choices))

# link formation by iterative weighted random selection:
@tf.function
def gen_topo( adj_matrix, prox_matrix, nnodes, tnedges, nbatches, n, k, t0, p0, weighted ):
    # we can add edges in batches, but not all at once
    #  we want to avoid adding more than 50% more edges to a graph at a time
    #  otherwise, good structure doesn't develop
    batch_size = tf.cast(nnodes, dtype=tf.float32) // 4.
    nedges = 0.
    # add edges until we hit the target
    while nedges < tnedges:
        # weighted-randomly choose links to add
        weights = compute_weights(adj_matrix, prox_matrix, n, k, t0, p0)
        choices = choose(weights, batch_size)
        # update adjacency matrix with new links
        nchoices = tf.shape(choices, out_type=tf.int32)[0]
        update = to_update(adj_matrix, choices, nchoices, nnodes)
        adj_matrix = tf.math.logical_or(adj_matrix, update)
        # how many links did we actually add?
        new_nedges = tf.reduce_sum(tf.cast(adj_matrix, dtype=tf.float32)) // 2.
        nchoices = new_nedges - nedges
        nedges = new_nedges
        # decide how many links to try to add next time
        if nchoices <= 1. + batch_size / 8.: break
        elif nchoices < batch_size / 4.:
            batch_size = batch_size // 2.
        elif nchoices >= batch_size / 2.:
            batch_size = 3. * batch_size // 2.
        batch_size = tf.math.minimum(batch_size, tnedges - nedges)
        batch_size = tf.math.maximum(batch_size, 2.)
    if weighted:
        return compute_final_weights(adj_matrix, prox_matrix)
    return adj_matrix

# handle how arguments and parameters are exposed:
def set_params(
    nnodes, nedges,
    ndimension=3, com_density=3.,
    p_factor=2.e6, t_factor=30,
    deg_ex=2.0, nbatches=12.,
    boost=200., cloud_offset=.01,
    weighted=True, seed=906619958
):
    nnodes = tf.cast(np.clip(nnodes, 2, 1e7), dtype=tf.int32)
    nedges = tf.cast(np.clip(nedges, 2, 1e8), dtype=tf.float32)
    ndimension = tf.cast(np.clip(ndimension + 1, 3, 12), dtype=tf.int32)
    com_density = tf.cast(np.clip(com_density, .5, 20), dtype=tf.float32)
    com_density = tf.cos(3.1416/2./com_density)
    deg_ex = tf.cast(np.clip(deg_ex, -10, 10), dtype=tf.float32)
    boost = tf.cast(np.clip(boost, 1, 1e12), dtype=tf.float32)
    boost = boost / (boost + 1.)
    t_factor = t_factor = tf.cast(np.clip(t_factor, 1, 1e12), dtype=tf.float32)
    t_factor = 1. / (t_factor + 1.)
    p_factor = tf.cast(np.clip(p_factor, 1, 1e12), dtype=tf.float32)
    p_factor = 1. / (p_factor + 1.)
    nbatches = tf.cast(np.clip(nbatches, 10, 100), dtype=tf.float32)
    cloud_offset = tf.cast(np.clip(cloud_offset, 0., 1.), dtype=tf.float32)
    tf.random.set_seed(seed)
    # set up the matrices
    adj_matrix = tf.zeros((nnodes, nnodes), dtype=tf.bool)
    prox_matrix = gen_cloud( nnodes, ndimension, cloud_offset, com_density )
    return [ adj_matrix, prox_matrix,
             nnodes, nedges, nbatches,
             deg_ex, boost, t_factor, p_factor, weighted ]

# recast the adjacency matrix tensor as a networkX graph:
def get_topo(matrix):
    matrix = matrix.numpy().astype(float)
    return nx.from_numpy_matrix(matrix)

# sample usage:
# nodes = 15000
# topo_params = {
#     'nnodes': nodes,
#     'nedges': 3*nodes,
#     'ndimension': 6,      # higher values prioritize interconnectedness
#     'cloud_offset': 0.18, # higher values prioritize more dynamic community sizes
#     'com_density': 4.,    # higher values prioritize smaller, more dense communities
#     'p_factor': 5.e6,     # higher values prioritize more modular communities
#     't_factor': 5.,       # higher values prioritize deep clustering
#     'deg_ex': 3.,
#     'seed': 906619958 }
# topo_params = set_params(**topo_params)

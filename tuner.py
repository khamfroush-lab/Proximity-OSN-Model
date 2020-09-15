import networkx as nx
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
import time
import IPython
from collections import Counter

import model

# compare two distributions using anderson-darling statistics
def compare_dists(a, b):
    try:
        stat = stats.anderson_ksamp([a, b])[0]
    except UserWarning: pass
    n = len(a) + len(b)
    stat = stat / ((n*n) / (n-1)) # normalize for n
    stat = stat / 0.507 + 0.1 # normalize to ~(0,1)
    return stat

# converts a difference into a loss; linear interpolation on:
#    actual | limit[0]  target  limit[1]
#      loss | bias[0]    0      bias[1]
def bounded_biased_loss(limit, bias, target, actual):
    if target == actual:
        return 0
    side = int(actual > target)
    return bias[side] * (1 - (actual - limit[side]) / (target - limit[side]))

# set up a log normal distribution with a and b at +1/ent and -1/ent
#    std deviations away from the mean, respectively 
def quick_lognorm(a, b):
    (a, b) = sorted(np.log([a,b]))
    return lambda ent = 1.0: np.random.lognormal((a+b)/2, ent * (b-a)/2)

# compute all relevant metrics on G
def metric_signature(G):
    deg_dist = np.array(list(G.degree()), dtype=int)[:, 1]
    spls = nx.all_pairs_shortest_path_length(G)
    spls = np.array(list(spls))[:, 1]
    spl_dist = []
    for n in spls:
        spl_dist += [item[1] for item in n.items()]
    coms = nx.algorithms.community.modularity_max.greedy_modularity_communities(G)
    com_dist = [len(l) for l in list(coms)]
    return {
        'nodes': nx.number_of_nodes(G),
        'edges': nx.number_of_edges(G),
        'assort': nx.degree_pearson_correlation_coefficient(G),
        'avgcc': nx.average_clustering(G),
        'module': nx.algorithms.community.quality.coverage(G, coms),
        'deg_dist': np.array(sorted(deg_dist)),
        'spl_dist': np.array(sorted(spl_dist)),
        'com_dist': np.array(sorted(com_dist))
    }

# heuristically compute a loss vector from the metrics
def compute_loss(target, actual):
    weights = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5])
    losses = np.array([
        bounded_biased_loss(
            (0.75 * target['nodes'], 1.25 * target['nodes']),
            (1.0, 0.01), target['nodes'], actual['nodes']),
        bounded_biased_loss(
            (0.75 * target['edges'], 1.25 * target['edges']),
            (1.0, 0.01), target['edges'], actual['edges']),
        bounded_biased_loss(
            (-1.0, 1.0), (1.0, 0.5),
            target['assort'], actual['assort']),
        bounded_biased_loss(
            (0.0, 1.0), (1.0, 0.5),
            np.sqrt(target['avgcc']), np.sqrt(actual['avgcc'])),
        bounded_biased_loss(
            (0.0, 1.0), (1.0, 0.5),
            target['module'], actual['module']),
        compare_dists(target['deg_dist'], actual['deg_dist']),
        compare_dists(target['spl_dist'], actual['spl_dist']),
        compare_dists(target['com_dist'], actual['com_dist'])
    ])
    return weights * losses

max_dim = 12

# determine good values for the non-optimized parameters
def find_defaults(A):
    return {
        'nodes': round(1.05 * A.number_of_nodes()),
        'edges': round(1.05 * A.number_of_edges()),
        'degree_max': round(1.1 * max(np.array(A.degree, dtype='int')[:, 1]))
    }

# pyplot live readout
def plt_readout(gen, population):
    plt.figure(figsize=(15,4))
    losses = [individual['t_loss'] for individual in population]
    losses = sorted(losses)[:int(len(losses)/2)]
    plt.hist(losses, bins = 15, histtype = 'step')
    IPython.display.clear_output(wait=False)
    print('population: ', len(population))
    print('least loss: ', population[0]['t_loss'])
    plt.show()

def plt_readout_advanced(gen, population):
    m = 0
    plt.figure(figsize=(30,4))
    for metric in population[0]['metrics']:
        plt.hist(
            np.transpose([unit['losses'] for unit in population])[m],
            label = metric, bins = 10, histtype = 'step'
        )
        m = m + 1
    plt.xscale('log')
    plt.legend(loc='upper right')
    p = 0
    fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize=(30,8))
    ax = ax.flatten()
    for param in population[0]['params']:
        ax[p].hist(
            np.transpose([
                [v for _, v in unit['params'].items()]
                for unit in population])[p],
            label = param, bins = 10, histtype = 'step',
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][p]
        )
        ax[p].legend(loc='upper right')
        p = p + 1
    IPython.display.clear_output(wait=False)
    print(gen, population[0]['t_loss'])
    print()
    plt.show()

# define a probabilistic search space for the parameters
space = {
    'dimension': lambda: np.random.choice([3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 10, 12]),
    'pref_exp': quick_lognorm(1.5, 3.0),
    'intg_exp': quick_lognorm(0.3, 0.6),
    'loner_boost': quick_lognorm(100, 1000),
    'prox_exp': quick_lognorm(2.0, 6.0),
    'prox_scale': quick_lognorm(0.2, 0.7),
    'prox_floor': quick_lognorm(1e-5, 5e-4)
}

# sample a novel individual from the search space
def synthesize():
    return {param: d() for param, d in space.items()}

# define a heuristic matrix to determine the relative likelihood
#   that each parameter is well-set, given the loss vector
blame = np.array([
   # i   k   a   c   m   d   s   o
    [0., 0., 0., .5, .5, 0., .5, .5], # dimension
    [0., 1., 0., 0., 0., 1., 0., .2], # preferential_exponent
    [0., 0., 1., 0., 0., .8, 1., 0.], # integrative_exponent
    [1., 0., 0., 0., 0., 0., 0., 0.], # loner_boost
    [0., 0., 0., 1., .5, 0., 0., 1.], # proximity_scale
    [0., 0., 0., .5, .5, .1, .2, 1.], # proximity_exponent
    [.2, 0., 0., .4, 1., .1, .2, .2]  # proximity_floor
]) + 0.1

# build a parameter set out into a topology and analyze it 
def grow(defaults, params, target):
    G = model.gen_topology(**defaults, **params)
    metrics = metric_signature(G)
    losses = compute_loss(target, metrics)
    metrics['deg_dist'] = Counter(metrics['deg_dist'])
    metrics['spl_dist'] = Counter(metrics['spl_dist'])
    metrics['com_dist'] = Counter(metrics['com_dist'])
    t_loss = np.sum(losses)
    blames = np.matmul(blame, losses) * t_loss
    return {
        'params': params, 'metrics': metrics,
        'losses': losses, 'blames': blames,
        't_loss': t_loss, 'graph': G
    }

# given a set of parents and some information about the population,
#   synthesize a new individual by smart genetic crossover
def crossover(parents, blame_avgs, blame_devs):
    offspring = {}
    p = 0
    # as loss diminishes, reduce the randomness of the crossover as well
    ent = 0.1 + np.sqrt(np.min([parent['t_loss'] for parent in parents])) / 2.0
    # set the selection pressures higher for parameters that are
    #   less well set in the population vs others based on blame
    pressure = 1.0 - blame_avgs / np.linalg.norm(blame_avgs)
    # turn the blames into relative fitnesses by comparing against the population
    rfits = ([parent['blames'] for parent in parents] - blame_avgs) / blame_devs
    # set each parameter independently
    for param in parents[0]['params']:
        # only sufficiently fit genes from the parents may be passed on 
        fit_genes = np.where(rfits[:,p] < pressure[p])[0]
        # select n genes from the pool to be combined
        n = min(fit_genes.size, np.random.choice([0, 1, 1, 1, 2, 2, 2]))
        fit_genes = np.random.choice(fit_genes, n, replace=False)
        # if no genes are available, synthesize a new one
        if fit_genes.size == 0:
            v = space[param]()
            offspring[param] = v
        # if one gene is available, mutate it
        elif fit_genes.size == 1:
            v = parents[fit_genes[0]]['params'][param] * quick_lognorm(0.9, 1/0.9)(ent)
            offspring[param] = v
        # if two are available, sample from a lognorm between them
        elif fit_genes.size >= 2:
            v = quick_lognorm(
                parents[fit_genes[0]]['params'][param],
                parents[fit_genes[1]]['params'][param] )(ent)
            offspring[param] = v
        p = p + 1
    offspring['dimension'] = int(np.clip(offspring['dimension'], 1, max_dim))
    return offspring

# attempt to achieve a set of target stats by using a genetic algorithm
#   to optimize the parameters of the generative model to match
def imitate(
    target, defaults = None, population = [],
    min_pop = 5, max_pop = 100, champion = (1, 0.02),
    generations = 200, readout = plt_readout
):
    # synthesize an intitial population
    while len(population) < min_pop:
        population.append(grow(defaults, synthesize(), target))
    try:
        gen = 0
        readout(gen, population)
        t = time.time()
        while gen < generations:
            # eliminate excess population, highest loss first:
            population = sorted(population, key = lambda unit: unit['t_loss'])
            population = population[:min(len(population), max_pop-1)]
            # determine the number of champions based on the population
            champions = int(champion[0] + champion[1] * len(population))
            # compute the population blame stats
            blame_avgs = np.average([unit['blames'] for unit in population], axis = 0)
            blame_devs = np.std([unit['blames'] for unit in population], axis = 0)
            ps = np.array([1 / unit['t_loss'] for unit in population])
            ps = ps / np.sum(ps)
            # select three parents
            parents = [
                population[np.random.randint(champions)], # a champion
                np.random.choice(np.array(population), p = ps), # one weighted by fitness
                np.random.choice(np.array(population)) # one unweighted
            ]
            # crossover to create a new individual 
            population.append(
                grow(defaults, crossover(parents, blame_avgs, blame_devs), target)
            )
            gen = gen + 1
            if time.time() - t > 3:
                readout(gen, population)
                t = time.time()
    except: pass # on error, we just dump the population db so nothing is lost
    return population

# attempt to replicate a specific target topology
def replicate(target, **params):
    target = nx.Graph(max(nx.connected_component_subgraphs(target), key=len))
    defaults = find_defaults(target)
    target = metric_signature(target)
    return imitate(target, defaults, **params)

# make another graph like a replica
def duplicate(target):
    return model.gen_topology(**find_defaults(target['graph']), **target['params'])
# Proximity-OSN-Model
This repository contains an implementation of the modeled described in the paper "A Proximity-Based Generative Model for Online Social Network Topologies" accepted for publication through the IEEE ICNC 2020 conference. The implementation is done in Python. This repository is provided as supplementary content for the referenced research paper.

## Basic usage:
To generate a random OSN-like community network:
```
import model
G = model.gen_topology()
```

You can use `tuner.metric_signature()` to measure an array of stats about any social network.
```
import tuner
tuner.metric_signature(G)
 # { 'nodes': 968,
 #   'edges': 3983,
 #   'assort': 0.2990160014933935,
 #   'avgcc': 0.27937334667139796,
 #   'module': 0.7865930203364299,
 #   'deg_dist': array([  1,   1,   1,  ..., 62,  80, 101]),
 #   'spl_dist': array([ 0,  0,  0, ..., 12, 12, 12]),
 #   'com_dist': array([  2,   2,   2,   ..., 116, 118, 203]) }
```

### Using the tuner:
```
import model
import tuner
population = tuner.replicate( your_sample_topology )
# let it run...

# now we have our first imitation:
copy1 = population[0]['graph']

# to make more:
copies = [tuner.duplicate(copy1) for n in range(10)]
```

#### Tuner monitoring:
The tuner will run continuously until it meets it's generation limit (by default 200 iterations). On keyboard interrupt (i+i on IPython) or other exception, the tuner safely returns the latest population. The tuner has a live readout which provides information on the population size and losses. If the population leader reaches a satisfactorily low level of loss, it is possible to interrupt it early and extract the data.

*Note: The live readout from the tuner is designed to run in an IPython cell and may not function in other environments*
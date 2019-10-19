# Proximity-OSN-Model
This repository contains an implementation of the modeled described in the paper "A Proximity-Based Generative Model for Online Social Network Topologies" accepted for publication through the IEEE ICNC 2020 conference. The implementation is done in Python. This repository is provided as supplementary content for the referenced research paper.

## Basic usage:
```
import model
import tuner

defaults = tuner.find_defaults( your_sample_topology )

# allow to run until loss is satisfactory:
population = tuner.replicate( your_sample_topology, defaults )

# now we have our first copy:
copy1 = population[0]['graph']

# to make more:
copy2 = model.gen_topology(**defaults, **population[0]['params'])
```

### Tuner monitoring:
The tuner will run continuously until it meets it's generation limit (by default 200 iterations). On keyboard interrupt (i+i on IPython) or other exception, the tuner safely returns the latest population. The tuner has a live readout which provides information on the population size and losses. If the population leader reaches a satisfactorily low level of loss, it is possible to interrupt it early and extract the data.

*Note: The live readout from the tuner is designed to run in an IPython cell and may not function in other environments*
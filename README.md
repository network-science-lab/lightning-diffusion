# Lightning Diffusion

Lightning fast network diffusion library

## API

### Methods
* `step()` - Performs a single step of the diffusion process.
* `run(n_iter: int = -1, verbose: bool=True)` - Runs multiple steps of the diffusion process. If `n_iter` < 0 - runs until termination.
* `reset()` - Resets the environment to an initial state, allowing for another simulation.

### Attributes
* `state: list` - current state of each node
* `iteration: int` - number of current iteration
* `terminated: bool` - flag signalling the termination of the diffusion model
* `state_space: tuple[str]` - Available node states
* `state_summary: dict[str, int]` - Current state summary - how many nodes are at each state

## Supported Models

### IndependenceCascadeModel

##### Example
```python
import networkx as nx

from lightning_diffusion.models import IndependentCascadeModel

graph = nx.barabasi_albert_graph(1000, 5)
model = IndependentCascadeModel(
    graph,
    infection_probability=0.01,
    initially_infected=0.1,
)

model.run(n_iter=5)

model.state_summary
>>> {'susceptible': 889, 'infected': 0, 'recovered': 111}
```


## Installation
```bash
pip install git+https://github.com/network-science-lab/lightning-diffusion
```

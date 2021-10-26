# torch-imle

Small and self-contained PyTorch library implementing the I-MLE gradient estimator proposed in the NeurIPS 2021 paper [Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions.](https://arxiv.org/abs/2106.01798)

## Example

Implicit MLE (I-MLE) wraps a black-box combinatorial solver, such as Dijkstra's algorithm, and:
1. Uses [Perturb-and-MAP](https://home.ttic.edu/~gpapan/research/perturb_and_map/) to transform the solver in an exponential family distribution we can sample from, and
2. Following the MLE for exponential family distributions (e.g. see [Murphy's book, Sect. 9.2.4](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf)), uses the difference between the sufficient statistics of the target distribution and their expectation according to the model to compute the gradient of the log-likelihood.

For example, let's consider this map, and the task is to find the shortest path from the top-left to the bottom-right corner of the map:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/map.png" width=600>

Here is what happens when we use a Sum-of-Gamma noise distribution to obtain a distribution over paths, using Dijkstra's algorithm:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/paths.gif" width=600>

## Gradients

Assuming the gold map is actually flat, and this is the gold shortest path:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/gold.png" width=600>

Here are the gradients of the Hamming loss between the inferred shortest path and the gold one wrt the map weights, produced by I-MLE, which can be used for learning the optimal map weights:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/gradients.gif" width=600>

## Code

Using this library is extremely easy -- see [this example](https://github.com/uclnlp/torch-imle/blob/main/annotation-cli.py) as a reference. Assuming we have a method that implements a black-box combinatorial solver such as Dijkstra's algorithm:

```python
import numpy as np

import torch
from torch import Tensor

def torch_solver(weights_batch: Tensor) -> Tensor:
    weights_batch = weights_batch.detach().cpu().numpy()
    y_batch = np.asarray([solver(w) for w in list(weights_batch)])
    return torch.tensor(y_batch, requires_grad=False)
```

We can obtain the corresponding distribution and gradients in this way:

```python
import numpy as np

import torch
from torch import Tensor

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
noise_distribution = SumOfGammaNoiseDistribution(k=k, nb_iterations=100)

def torch_solver(weights_batch: Tensor) -> Tensor:
    weights_batch = weights_batch.detach().cpu().numpy()
    y_batch = np.asarray([solver(w) for w in list(weights_batch)])
    return torch.tensor(y_batch, requires_grad=False)

imle_solver = imle(torch_solver,
                   target_distribution=target_distribution,
                    noise_distribution=noise_distribution,
                    nb_samples=10,
                    input_noise_temperature=input_noise_temperature,
                    target_noise_temperature=target_noise_temperature)
```

Or, alternatively, using a simple function annotation:

```python
@imle(target_distribution=target_distribution,
      noise_distribution=noise_distribution,
      nb_samples=10,
      input_noise_temperature=input_noise_temperature,
      target_noise_temperature=target_noise_temperature)
def imle_solver(weights_batch: Tensor) -> Tensor:
    return torch_solver(weights_batch)
```

## Reference

```bibtex
@inproceedings{niepert21imle,
  author    = {Mathias Niepert and
               Pasquale Minervini and
               Luca Franceschi},
  title     = {Implicit {MLE:} Backpropagating Through Discrete Exponential Family
               Distributions},
  booktitle = {NeurIPS},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR}},
  year      = {2021}
}
```
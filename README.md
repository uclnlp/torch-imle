# torch-imle

Concise and self-contained PyTorch library implementing the I-MLE gradient estimator proposed in the NeurIPS 2021 paper [Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions.](https://arxiv.org/abs/2106.01798)

## Example

Implicit MLE (I-MLE) makes it possible to integrete discrete combinatorial optimization algorithms, such as Dijkstra's algorithm or general integer linear programs, into standard deep learning architectures. The core idea of I-MLE is that it defines an *implicit* maximum likelihood objective whose gradients are used to update upstream parameters of the model. Every instance of I-MLE requires two ingredients:
1. An ability to approximately sample from a complex and intractable distribution. For this we use perturb and MAP [Perturb-and-MAP](https://home.ttic.edu/~gpapan/research/perturb_and_map/) (aka the Gumbel-max trick) and propose a novel family of noise perturbations tailored to the problem at hand. 
2. I-MLE reduces the KL divergence between the current distribution and the empirical distribution. Since in our setting, we do not have access to the empirical distribution and, therefore, have to design surrogate empirical distributions. Here we propose two families of surrogate distributions which are widely applicable and work well in practice. 

For example, let's consider a map from a simple game where the task is to find the shortest path from the top-left to the bottom-right corner:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/map.png" width=600>

Here is what happens when we use the proposed sum-of-gamma noise distribution to obtain a distribution over paths, using Dijkstra's algorithm:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/paths.gif" width=600>

## Gradients

Assuming the gold map is actually flat, and this is the gold shortest path:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/gold.png" width=600>

Here are the gradients of the Hamming loss between the inferred shortest path and the gold one wrt the map weights, produced by I-MLE, which can be used for learning the optimal map weights:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/figures/gradients.gif" width=600>

Note that minimising the downstream training objective (the Hamming loss between the inferred and the gold paths) using gradient descent will decrease the weights of the cells on the diagonal of the map, while increasing the cost of the paths sampled by the model, pushing the model towards the gold path.

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

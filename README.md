# torch-imle

PyTorch library implementing the I-MLE gradient estimator proposed in the NeurIPS 2021 paper [Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions.](https://arxiv.org/abs/2106.01798)

### Example

Implicit MLE (I-MLE) wraps a black-box combinatorial solver, such as Dijkstra's algorithm, and:
1. Uses [Perturb-and-MAP](https://home.ttic.edu/~gpapan/research/perturb_and_map/) to transform the solver in an exponential family distribution we can sample from, and
2. Following MLE for the exponential family (e.g. see [Murphy's book, Sect. 9.2.4](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf)), uses the sufficient statistics of the target distribution and their expectation according to the model to compute the gradient of the log-likelihood.

Here is what happens when we use a Sum-of-Gamma noise distribution to obtain a distribution over paths, using Dijkstra's algorithm:

<img src="https://raw.githubusercontent.com/uclnlp/torch-imle/main/animations/paths.gif" width=600>


### Reference

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
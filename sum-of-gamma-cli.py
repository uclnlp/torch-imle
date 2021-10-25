#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import math
import numpy as np
import torch

from torch import Tensor
from torch.distributions.gamma import Gamma

import matplotlib.pyplot as plt


def sample_sum_of_gammas(nb_samples: int,
                         k: np.ndarray,
                         nb_iterations: int = 10):
    """These should be the epsilons from Th, 1, """
    k_size = k.shape[0]
    samples = np.zeros((nb_samples, k_size), dtype='float')
    for i in range(1, nb_iterations + 1):
        print('XXX', 1. / k, k / i)
        gs = np.random.gamma(1. / k, k / i, size=[nb_samples, k_size])
        samples = samples + gs
    samples = ((samples - math.log(nb_iterations)) / k)
    return samples


def sample_sum_of_gammas_torch(batch_size: int,
                               k: Tensor,
                               nb_iterations: int = 10):
    nb_samples = k.shape[0]
    samples = torch.zeros((batch_size, nb_samples))
    for i in range(1, nb_iterations + 1):
        gamma = Gamma(1. / k, i / k)
        samples = samples + gamma.sample(sample_shape=torch.Size([batch_size]))
    samples = (samples - math.log(nb_iterations)) / k
    return samples


def main(argv):
    # samples = sample_sum_of_gammas(8192, k=np.ones(20) * 20, nb_iterations=100)
    # print(samples)

    # with torch.inference_mode():
    #     samples = sample_sum_of_gammas_torch(1024, k=torch.ones(20) * 20, nb_iterations=100).cpu().numpy()

    from imle.noise import SumOfGammaNoiseDistribution
    distribution = SumOfGammaNoiseDistribution(k=20.0, nb_iterations=1000)
    with torch.inference_mode():
        samples = distribution.sample(shape=torch.Size([8192, 20])).cpu().numpy()

    count, bins, ignored = plt.hist(np.sum(samples, axis=1), 32, density=True)

    mu, beta = 0.0, 1.0
    y = (1 / beta) * np.exp(-(bins - mu) / beta) * np.exp(-np.exp(-(bins - mu) / beta))

    plt.plot(bins, y, linewidth=2, color='r')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])

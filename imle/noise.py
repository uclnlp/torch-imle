# -*- coding: utf-8 -*-

import math

import torch
from torch import Tensor, Size
from torch.distributions.gamma import Gamma

from abc import ABC, abstractmethod

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class BaseNoiseDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self,
               shape: Size) -> Tensor:
        raise NotImplementedError


class SumOfGammaNoiseDistribution(BaseNoiseDistribution):
    r"""
    Creates a generator of samples for the Sum-of-Gamma distribution [1], parameterized
    by :attr:`k`, :attr:`nb_iterations`, and :attr:`device`.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)

    Example::

        >>> import torch
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=5, nb_iterations=100)
        >>> noise_distribution.sample(torch.Size([5]))
        tensor([ 0.2504,  0.0112,  0.5466,  0.0051, -0.1497])

    Args:
        k (float): k parameter -- see [1] for more details.
        nb_iterations (int): number of iterations for estimating the sample.
        device (torch.devicde): device where to store samples.
    """
    def __init__(self,
                 k: float,
                 nb_iterations: int = 10,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.k = k
        self.nb_iterations = nb_iterations
        self.device = device

    def sample(self,
               shape: Size) -> Tensor:
        samples = torch.zeros(size=shape, device=self.device)
        for i in range(1, self.nb_iterations + 1):
            concentration = torch.tensor(1. / self.k, device=self.device)
            rate = torch.tensor(i / self.k, device=self.device)

            gamma = Gamma(concentration=concentration, rate=rate)
            samples = samples + gamma.sample(sample_shape=shape).to(self.device)
        samples = (samples - math.log(self.nb_iterations)) / self.k
        return samples.to(self.device)

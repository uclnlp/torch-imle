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

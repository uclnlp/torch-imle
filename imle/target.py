# -*- coding: utf-8 -*-

from torch import Tensor
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)


class BaseTargetDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:
        raise NotImplementedError


class TargetDistribution(BaseTargetDistribution):
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:
        theta_prime = self.alpha * theta + self.beta * dy
        return theta_prime

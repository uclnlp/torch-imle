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
    r"""
    Creates a generator of target distributions parameterized by :attr:`alpha` and :attr:`beta`.

    Example::

        >>> import torch
        >>> target_distribution = TargetDistribution(alpha=1.0, beta=1.0)
        >>> target_distribution.params(theta=torch.tensor([1.0]), dy=torch.tensor([1.0]))
        tensor([2.])

    Args:
        alpha (float): weight of the initial distribution parameters theta
        beta (float): weight of the downstream gradient dy
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:
        theta_prime = self.alpha * theta - self.beta * dy
        return theta_prime

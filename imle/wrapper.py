# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor

from imle.noise import BaseNoiseDistribution
from imle.target import BaseTargetDistribution, TargetDistribution

from typing import Callable, Optional

import logging

logger = logging.getLogger(__name__)


def imle(function: Callable[[Tensor], Tensor] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         nb_samples: int = 1,
         input_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0):
    r"""Turns a black-box combinatorial solver in an Exponential Family distribution via Perturb-and-MAP and I-MLE [1].

    The input function (solver) needs to return the solution to the problem of finding a MAP state for a constrained
    exponential family distribution -- this is the case for most black-box combinatorial solvers [2]. If this condition
    is violated though, the result would not hold and there ismno guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)
    [2] Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, Michal Rolínek - Differentiation of Blackbox
    Combinatorial Solvers. ICLR 2020 (https://arxiv.org/abs/1912.02175)

    Example::

        >>> from imle.wrapper import imle
        >>> from imle.target import TargetDistribution
        >>> from imle.noise import SumOfGammaNoiseDistribution
        >>> target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
        >>>       input_noise_temperature=input_noise_temperature, target_noise_temperature=5.0)
        >>> def imle_solver(weights_batch: Tensor) -> Tensor:
        >>>     return torch_solver(weights_batch)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        target_distribution (Optional[BaseTargetDistribution]): factory for target distributions
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise sammples
        input_noise_temperature (float): noise temperature for the input distribution
        target_noise_temperature (float): noise temperature for the target distribution
    """
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(imle,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 nb_samples=nb_samples,
                                 input_noise_temperature=input_noise_temperature,
                                 target_noise_temperature=target_noise_temperature)

    @functools.wraps(function)
    def wrapper(input: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input: Tensor, *args):
                # [BATCH_SIZE, ...]
                input_shape = input.shape

                batch_size = input_shape[0]
                instance_shape = input_shape[1:]

                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_input_shape = [batch_size, nb_samples] + list(instance_shape)

                if noise_distribution is None:
                    noise = torch.zeros(size=perturbed_input_shape)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_input_shape))

                input_noise = noise * input_noise_temperature

                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_input_3d = input.view(batch_size, 1, -1).repeat(1, nb_samples, 1).view(perturbed_input_shape)
                perturbed_input_3d = perturbed_input_3d + input_noise

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_input_2d = perturbed_input_3d.view([-1] + perturbed_input_shape[2:])

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_output = function(perturbed_input_2d)
                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_output = perturbed_output.view(perturbed_input_shape)

                ctx.save_for_backward(input, noise, perturbed_output)

                # [BATCH_SIZE * N_SAMPLES, ...]
                res = perturbed_output.view([-1] + perturbed_input_shape[2:])
                return res

            @staticmethod
            def backward(ctx, dy):
                # input: [BATCH_SIZE, ...]
                # noise: [BATCH_SIZE, N_SAMPLES, ...]
                # perturbed_output_3d: # [BATCH_SIZE, N_SAMPLES, ...]
                input, noise, perturbed_output_3d = ctx.saved_variables

                input_shape = input.shape
                batch_size = input_shape[0]

                # dy is [BATCH_SIZE * N_SAMPLES, ...]
                dy_shape = dy.shape
                # noise is [BATCH_SIZE, N_SAMPLES, ...]
                noise_shape = noise.shape

                # [BATCH_SIZE * NB_SAMPLES, ...]
                input_2d = input.view(batch_size, 1, -1).repeat(1, nb_samples, 1).view(dy_shape)
                target_input_2d = target_distribution.params(input_2d, dy)
                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_input_3d = target_input_2d.view(noise_shape)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_noise = noise * target_noise_temperature

                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_target_input_3d = target_input_3d + target_noise

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_target_input_2d = perturbed_target_input_3d.view(dy_shape)

                # [BATCH_SIZE * N_SAMPLES, ...]
                target_output_2d = function(perturbed_target_input_2d)
                # [BATCH_SIZE, N_SAMPLES, ...]
                target_output_3d = target_output_2d.view(noise_shape)

                # [BATCH_SIZE, ...]
                gradient = (perturbed_output_3d - target_output_3d).mean(axis=1)
                return gradient

        return WrappedFunc.apply(input, *args)
    return wrapper

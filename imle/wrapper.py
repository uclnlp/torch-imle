# -*- coding: utf-8 -*-

import functools

import torch
from torch import Tensor, Size

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
                # [BATCH_SIZE, ..]
                input_shape = input.shape

                # [N_SAMPLES, BATCH_SIZE, ..]
                perturbed_input_shape = [nb_samples] + list(input_shape)

                if noise_distribution is None:
                    noise = torch.zeros(size=perturbed_input_shape)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_input_shape))

                input_noise = noise * input_noise_temperature

                # [N_SAMPLES, BATCH_SIZE, ..]
                perturbed_input_3d = input.unsqueeze(0) + input_noise

                # [N_SAMPLES * BATCH_SIZE, ..]
                perturbed_input_2d = perturbed_input_3d.view([-1] + perturbed_input_shape[2:])

                # [N_SAMPLES * BATCH_SIZE, ..]
                perturbed_output = function(perturbed_input_2d)
                # [N_SAMPLES, BATCH_SIZE, ..]
                perturbed_output = perturbed_output.view(perturbed_input_shape)

                ctx.save_for_backward(input, noise, perturbed_output)

                # [BATCH_SIZE, ..]
                res = perturbed_output.mean(0)
                return res

            @staticmethod
            def backward(ctx, dy):
                # input: [BATCH_SIZE, ..]
                # noise: [N_SAMPLES, BATCH_SIZE, ..]
                # perturbed_output: # [N_SAMPLES, BATCH_SIZE, ..]
                input, noise, perturbed_output = ctx.saved_variables

                # [BATCH_SIZE, ..]
                target_input = target_distribution.params(input, dy)

                target_noise = noise * target_noise_temperature

                # [N_SAMPLES, BATCH_SIZE, ..]
                perturbed_target_input_3d = target_input.unsqueeze(0) + target_noise

                # [N_SAMPLES, BATCH_SIZE, ..]
                perturbed_target_input_shape = list(perturbed_target_input_3d.shape)

                # [N_SAMPLES * BATCH_SIZE, ..]
                perturbed_target_input_2d = perturbed_target_input_3d.view([-1] + perturbed_target_input_shape[2:])

                # [N_SAMPLES * BATCH_SIZE, ..]
                target_output = function(perturbed_target_input_2d)
                # [N_SAMPLES, BATCH_SIZE, ..]
                target_output.view(perturbed_target_input_shape)

                gradient = - (perturbed_output.mean(0) - target_output.mean(0))
                return gradient

        return WrappedFunc.apply(input, *args)
    return wrapper

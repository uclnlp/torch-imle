#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np

import torch
from torch import nn, Tensor

from imle.wrapper import imle
from imle.target import TargetDistribution
from imle.noise import SumOfGammaNoiseDistribution

from solvers.dijkstra import get_solver

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import animation


class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()


def main(argv):
    neighbourhood_fn = "4-grid"
    solver = get_solver(neighbourhood_fn)

    grid_size = [16, 16]

    def torch_solver(weights_batch: Tensor) -> Tensor:
        weights_batch = weights_batch.detach().cpu().numpy()
        y_batch = np.asarray([solver(w) for w in list(weights_batch)])
        return torch.tensor(y_batch, requires_grad=False)

    with torch.inference_mode():
        weights_1 = np.ones(shape=grid_size, dtype=float)
        weights_1[1:12, 1:12] = 2

        weights_2 = np.ones(shape=grid_size, dtype=float)
        weights_2[1:4, 0] = 0
        weights_2[3, 0:3] = 0

        weights_1_batch = torch.tensor(weights_1).unsqueeze(0)
        weights_2_batch = torch.tensor(weights_2).unsqueeze(0)

        y_1_batch = torch_solver(weights_1_batch)
        y_2_batch = torch_solver(weights_2_batch)

    loss_fn = HammingLoss()

    def generate_distribution(input_noise_temperature: float = 5.0):
        weights_1 = np.ones(shape=[1] + grid_size, dtype=float)
        weights_1[0, 1:6, 0:12] = 100
        weights_1[0, 8:12, 1:] = 100
        weights_1[0, 14, 6:10] = 100

        weights_1_tensor = torch.tensor(weights_1)
        weights_1_params = nn.Parameter(weights_1_tensor, requires_grad=True)

        # print(weights_1_params[0])

        y_2_tensor = torch.tensor(y_2_batch.detach().cpu().numpy())

        target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        noise_distribution = SumOfGammaNoiseDistribution(k=grid_size[0] * 1.3, nb_iterations=100)

        imle_solver = imle(torch_solver,
                           target_distribution=target_distribution,
                           noise_distribution=noise_distribution,
                           nb_samples=10,
                           input_noise_temperature=input_noise_temperature,
                           target_noise_temperature=5.0)

        imle_y_tensor = imle_solver(weights_1_params)

        loss = loss_fn(imle_y_tensor, y_2_tensor)

        loss.backward()

        return weights_1, imle_y_tensor, y_2_tensor, weights_1_params

    def init_fwd():
        weights_1, imle_y_tensor, y_2_tensor, weights_1_params = generate_distribution(0.0)

        sns.set_theme()
        ax = sns.heatmap(imle_y_tensor[0].detach().cpu().numpy())

        ax.set_title(f'Distribution over paths -- input noise temperature: {0.0:.2f}')

    def animate_fwd(t):
        plt.clf()
        weights_1, imle_y_tensor, y_2_tensor, weights_1_params = generate_distribution(t * 0.1)

        sns.set_theme()
        ax = sns.heatmap(imle_y_tensor[0].detach().cpu().numpy())

        ax.set_title(f'Distribution over paths -- input noise temperature: {t * 0.1:.2f}')

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate_fwd, init_func=init_fwd, frames=80, repeat=False)

    anim.save('animations/paths.gif', writer='imagemagick', fps=8)

    # plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])

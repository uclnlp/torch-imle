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
    neighbourhood_fn = "8-grid"
    solver = get_solver(neighbourhood_fn)

    grid_size = [16, 16]

    def torch_solver(weights_batch: Tensor) -> Tensor:
        r"""
        Wrapper around the `solver` function, which implements Dijkstra's Shortest Path Algorithm.
        Note that I-MLE assumes that the solver solves a maximisation problem, but here the `solver` function solves
        a minimisation problem, by finding the path that minimises the total travel cost. As a consequence, here
        we flip the sign of the weights: `solver` will find the path with minimal total cost, while `torch_solver`
        will find the path that maximises the total gain (all weights are now negative).

        Args:
            weights_batch (Tensor): PyTorch tensor with shape [BATCH_SIZE, MAP_WIDTH, MAP_HEIGHT]
        """
        weights_batch = weights_batch.detach().cpu().numpy()
        y_batch = np.asarray([solver(- 1.0 * w) for w in list(weights_batch)])
        return torch.tensor(y_batch, requires_grad=False)

    with torch.inference_mode():
        weights = np.empty(shape=[1] + grid_size, dtype=float)
        weights.fill(-1)

        weights_batch = torch.tensor(weights)
        y_batch = torch_solver(weights_batch)

    loss_fn = HammingLoss()

    def generate_distribution(input_noise_temperature: float = 5.0):
        weights = np.empty(shape=[1] + grid_size, dtype=float)
        weights.fill(-1)

        weights[0, 1:6, 0:12] = -100
        weights[0, 8:12, 1:] = -100
        weights[0, 14:, 6:10] = -100

        weights_tensor = torch.tensor(weights)
        weights_params = nn.Parameter(weights_tensor, requires_grad=True)

        y_tensor = torch.tensor(y_batch.detach().cpu().numpy())

        target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        noise_distribution = SumOfGammaNoiseDistribution(k=grid_size[0] * 1.3, nb_iterations=100)

        # imle_solver = imle(torch_solver,
        #                    target_distribution=target_distribution,
        #                    noise_distribution=noise_distribution,
        #                    input_noise_temperature=input_noise_temperature,
        #                    target_noise_temperature=5.0)

        @imle(target_distribution=target_distribution, noise_distribution=noise_distribution,
              input_noise_temperature=input_noise_temperature, target_noise_temperature=5.0)
        def imle_solver(weights_batch: Tensor) -> Tensor:
            return torch_solver(weights_batch)

        imle_y_tensor = imle_solver(weights_params)
        loss = loss_fn(imle_y_tensor, y_tensor)
        loss.backward()

        return weights, imle_y_tensor, y_tensor, weights_params

    weights, imle_y_tensor, y_tensor, weights_params = generate_distribution(0.0)

    sns.set_theme()
    ax = sns.heatmap(weights[0])

    ax.set_title(f'Map')

    fig = ax.get_figure()
    fig.savefig("figures/map.png")

    plt.clf()

    sns.set_theme()
    ax = sns.heatmap(y_tensor[0].detach().cpu().numpy())

    ax.set_title(f'Gold Path')

    fig = ax.get_figure()
    fig.savefig("figures/gold.png")

    sampled_paths_lst = []

    def init_fwd():
        nonlocal sampled_paths_lst
        plt.clf()
        weights, imle_y_tensor, y_tensor, weights_params = generate_distribution(1.0)
        sampled_paths_lst += [imle_y_tensor[0].detach().cpu().numpy()]
        sns.set_theme()
        ax = sns.heatmap(np.mean(sampled_paths_lst, axis=0))
        ax.set_title(f'Sampled paths -- temperature 1.0, iteration: {0}')

    def animate_fwd(i):
        nonlocal sampled_paths_lst
        plt.clf()
        weights, imle_y_tensor, y_tensor, weights_params = generate_distribution(1.0)
        sampled_paths_lst += [imle_y_tensor[0].detach().cpu().numpy()]
        sns.set_theme()
        ax = sns.heatmap(np.mean(sampled_paths_lst, axis=0))
        ax.set_title(f'Sampled paths -- temperature 1.0, iteration: {i + 1}')

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate_fwd, init_func=init_fwd, frames=100, repeat=False)

    anim.save('figures/paths.gif', writer='imagemagick', fps=8)

    plt.clf()

    gradients_lst = []

    def init_grad():
        nonlocal gradients_lst
        plt.clf()
        weights, imle_y_tensor, y_tensor, weights_params = generate_distribution(1.0)
        gradients_lst += [weights_params.grad[0].detach().cpu().numpy()]
        sns.set_theme()
        ax = sns.heatmap(np.mean(gradients_lst, axis=0))
        ax.set_title(f'Gradient -- temperature 1.0, iteration: {0}')

    def animate_grad(t):
        nonlocal gradients_lst
        plt.clf()
        weights, imle_y_tensor, y_tensor, weights_params = generate_distribution(1.0)
        gradients_lst += [weights_params.grad[0].detach().cpu().numpy()]
        sns.set_theme()
        ax = sns.heatmap(np.mean(gradients_lst, axis=0))
        ax.set_title(f'Gradient -- temperature 1.0, iteration: {0}')

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate_grad, init_func=init_grad, frames=100, repeat=False)

    anim.save('figures/gradients.gif', writer='imagemagick', fps=8)


if __name__ == '__main__':
    main(sys.argv[1:])

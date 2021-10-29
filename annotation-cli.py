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

import multiprocessing
import ray
import random
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import animation


def set_seed(seed: int, is_deterministic: bool = True):
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def maybe_parallelize(function, arg_list):
    if ray.is_initialized():
        ray_fn = ray.remote(function)
        return ray.get([ray_fn.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]


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
        weights_batch = - 1.0 * weights_batch.detach().cpu().numpy()
        # y_batch = np.asarray([solver(w) for w in list(weights_batch)])
        y_batch = np.asarray(maybe_parallelize(solver, arg_list=list(weights_batch)))
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

        target_distribution = TargetDistribution(alpha=1.0, beta=10.0)
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

    # Inference

    weights, imle_y_tensor, y_tensor, weights_params = generate_distribution(0.0)

    sns.set_theme()
    ax = sns.heatmap(weights[0])

    ax.set_title(f'Map')

    fig = ax.get_figure()
    fig.savefig("figures/map.png")

    plt.clf()

    sampled_paths_lst = []
    for i in range(110):
        weights, imle_y_tensor, y_tensor, weights_params = generate_distribution(1.0)
        sampled_paths_lst += [imle_y_tensor[0].detach().cpu().numpy()]

    def init_fwd():
        nonlocal sampled_paths_lst
        plt.clf()
        sns.set_theme()
        ax = sns.heatmap(sampled_paths_lst[0]) #, vmin=0.0, vmax=1.0)
        ax.set_title(f'Sampled path -- temperature 1.0, iteration: {0}')

    def animate_fwd(i):
        nonlocal sampled_paths_lst
        plt.clf()
        sns.set_theme()
        ax = sns.heatmap(sampled_paths_lst[i]) #, vmin=0.0, vmax=1.0)
        ax.set_title(f'Sampled path -- temperature 1.0, iteration: {i}')

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate_fwd, init_func=init_fwd, frames=100, repeat=False)

    anim.save('figures/paths.gif', writer='imagemagick', fps=8)

    plt.clf()

    def init_fwd():
        nonlocal sampled_paths_lst
        plt.clf()
        sns.set_theme()
        ax = sns.heatmap(sampled_paths_lst[0]) #, vmin=0.0, vmax=1.0)
        ax.set_title(f'Distribution over paths -- temperature 1.0, iteration: {0}')

    def animate_fwd(i):
        nonlocal sampled_paths_lst
        plt.clf()
        sns.set_theme()
        ax = sns.heatmap(np.mean(sampled_paths_lst[:i + 1], axis=0)) #, vmin=0.0, vmax=1.0)
        ax.set_title(f'Distribution over paths -- temperature 1.0, iteration: {i}')

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate_fwd, init_func=init_fwd, frames=100, repeat=False)

    anim.save('figures/distribution.gif', writer='imagemagick', fps=8)

    plt.clf()


def learning(argv):
    set_seed(0)

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
        weights_batch = - 1.0 * weights_batch.detach().cpu().numpy()
        # y_batch = np.asarray([solver(w) for w in list(weights_batch)])
        y_batch = np.asarray(maybe_parallelize(solver, arg_list=list(weights_batch)))
        return torch.tensor(y_batch, requires_grad=False)

    for input_noise_temperature in [0.0, 1.0, 2.0, 5.0]:
        for target_noise_temperature in [0.0, 1.0, 2.0, 5.0]:
            for nb_samples in [1, 10, 100]:
                # Gradients
                true_weights = np.empty(shape=[1] + grid_size, dtype=float)
                true_weights.fill(-1)

                true_weights[0, 1:6, 0:12] = -100
                true_weights[0, 8:12, 1:] = -100
                true_weights[0, 14:, 6:10] = -100

                true_y = torch_solver(torch.tensor(true_weights)).detach()

                plt.clf()

                sns.set_theme()
                ax = sns.heatmap(np.copy(true_y[0].cpu().numpy()))

                ax.set_title(f'Gold path')

                fig = ax.get_figure()
                fig.savefig(f"figures/gold_int={input_noise_temperature}_tnt={target_noise_temperature}_ns={nb_samples}.png")

                weights = np.random.uniform(low=-0.1, high=0.1, size=[1] + grid_size)
                weights_tensor = torch.tensor(weights, dtype=torch.float)
                weights_params = nn.Parameter(weights_tensor, requires_grad=True)

                optimizer = torch.optim.Adam([weights_params], lr=0.005)

                evolving_weights_lst = []
                evolving_paths_lst = []

                loss_fn = HammingLoss()

                for t in range(1100):
                    target_distribution = TargetDistribution(alpha=1.0, beta=10.0)
                    noise_distribution = SumOfGammaNoiseDistribution(k=grid_size[0] * 1.3, nb_iterations=100)

                    @imle(target_distribution=target_distribution,
                          noise_distribution=noise_distribution,
                          input_noise_temperature=input_noise_temperature,
                          target_noise_temperature=target_noise_temperature,
                          nb_samples=nb_samples)
                    def imle_solver(weights_batch: Tensor) -> Tensor:
                        return torch_solver(weights_batch)

                    imle_y_tensor = imle_solver(weights_params)

                    evolving_weights_lst += [np.copy(weights_params[0].detach().cpu().numpy())]
                    evolving_paths_lst += [np.copy(imle_y_tensor[0].detach().cpu().numpy())]

                    loss = loss_fn(imle_y_tensor, true_y)

                    if t % 10:
                        print(f"Iteration: {t}\tLoss: {loss.item():.2f}")

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                def init_paths():
                    nonlocal evolving_paths_lst
                    plt.clf()
                    sns.set_theme()
                    ax = sns.heatmap(evolving_paths_lst[0]) #, vmin=0.0, vmax=1.0)
                    ax.set_title(f'Inferred path -- Input noise {input_noise_temperature}, '
                                 f'target noise {target_noise_temperature}, iteration: {0}')

                def animate_paths(i):
                    nonlocal evolving_paths_lst
                    plt.clf()
                    sns.set_theme()
                    ax = sns.heatmap(evolving_paths_lst[i * 10]) #, vmin=0.0, vmax=1.0)
                    ax.set_title(f'Inferred path -- Input noise {input_noise_temperature}, '
                                 f'target noise {target_noise_temperature}, iteration: {i}')

                fig = plt.figure()
                anim = animation.FuncAnimation(fig, animate_paths, init_func=init_paths, frames=100, repeat=False)

                anim.save(f'figures/learning_paths_int={input_noise_temperature}_tnt={target_noise_temperature}_ns={nb_samples}.gif',
                          writer='imagemagick', fps=8)

                def init_weights():
                    nonlocal evolving_paths_lst
                    plt.clf()
                    sns.set_theme()
                    ax = sns.heatmap(evolving_weights_lst[0]) #, vmin=0.0, vmax=1.0)
                    ax.set_title(f'Inferred weights -- Input noise {input_noise_temperature}, '
                                 f'target noise {target_noise_temperature}, iteration: {0}')

                def animate_weights(i):
                    nonlocal evolving_paths_lst
                    plt.clf()
                    sns.set_theme()
                    ax = sns.heatmap(evolving_weights_lst[i * 10]) #, vmin=0.0, vmax=1.0)
                    ax.set_title(f'Inferred weights -- Input noise {input_noise_temperature}, '
                                 f'target noise {target_noise_temperature}, iteration: {i}')

                fig = plt.figure()
                anim = animation.FuncAnimation(fig, animate_weights, init_func=init_weights, frames=100, repeat=False)

                anim.save(f'figures/learning_weights_int={input_noise_temperature}_tnt={target_noise_temperature}_ns={nb_samples}.gif',
                          writer='imagemagick', fps=8)

                plt.clf()


if __name__ == '__main__':
    ray.init(num_cpus=multiprocessing.cpu_count())

    main(sys.argv[1:])
    learning(sys.argv[1:])

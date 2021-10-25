# -*- coding: utf-8 -*-

import numpy as np
import heapq
import torch

from functools import partial

from solvers.utils import get_neighbourhood_func, maybe_parallelize
from collections import namedtuple

DijkstraOutput = namedtuple("DijkstraOutput",
                            [
                                "shortest_path",
                                "is_unique",
                                "transitions"
                            ])


def dijkstra(matrix, neighbourhood_fn="8-grid", request_transitions=False):
    x_max, y_max = matrix.shape
    neighbors_func = partial(get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max)

    costs = np.full_like(matrix, 1.0e10)
    costs[0][0] = matrix[0][0]
    num_path = np.zeros_like(matrix)
    num_path[0][0] = 1
    priority_queue = [(matrix[0][0], (0, 0))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))
    # retrieve the path
    cur_x, cur_y = x_max - 1, y_max - 1
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1
    while (cur_x, cur_y) != (0, 0):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[-1, -1] == 1

    if request_transitions:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
    else:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)


def get_solver(neighbourhood_fn):
    def solver(matrix):
        return dijkstra(matrix, neighbourhood_fn).shortest_path

    return solver


class ShortestPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, lambda_val, neighbourhood_fn="8-grid"):
        ctx.lambda_val = lambda_val
        ctx.neighbourhood_fn = neighbourhood_fn
        ctx.solver = get_solver(neighbourhood_fn)

        ctx.weights = weights.detach().cpu().numpy()
        ctx.suggested_tours = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(ctx.weights)))
        return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()
        weights_prime = np.maximum(ctx.weights + ctx.lambda_val * grad_output_numpy, 0.0)
        better_paths = np.asarray(maybe_parallelize(ctx.solver, arg_list=list(weights_prime)))
        gradient = -(ctx.suggested_tours - better_paths) / ctx.lambda_val
        return torch.from_numpy(gradient).to(grad_output.device), None, None

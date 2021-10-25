# -*- coding: utf-8 -*-

import itertools
import functools
import numpy as np

import ray


def neighbours_8(x, y, x_max, y_max):
    deltas_x = (-1, 0, 1)
    deltas_y = (-1, 0, 1)
    for (dx, dy) in itertools.product(deltas_x, deltas_y):
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def neighbours_4(x, y, x_max, y_max):
    for (dx, dy) in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def get_neighbourhood_func(neighbourhood_fn):
    if neighbourhood_fn == "4-grid":
        return neighbours_4
    elif neighbourhood_fn == "8-grid":
        return neighbours_8
    else:
        raise Exception(f"neighbourhood_fn of {neighbourhood_fn} not possible")


def edges_from_vertex(x, y, N, neighbourhood_fn):
    v = (x, y)
    neighbours = get_neighbourhood_func(neighbourhood_fn)(*v, x_max=N, y_max=N)
    v_edges = [
        (*v, *vn) for vn in neighbours if vertex_index(v, N) < vertex_index(vn, N)
    ]  # Enforce ordering on vertices
    return v_edges


def vertex_index(v, dim):
    x, y = v
    return x * dim + y


@functools.lru_cache(32)
def edges_from_grid(N, neighbourhood_fn):
    all_vertices = itertools.product(range(N), range(N))
    all_edges = [edges_from_vertex(x, y, N, neighbourhood_fn=neighbourhood_fn) for x, y in all_vertices]
    all_edges_flat = sum(all_edges, [])
    all_edges_flat_unique = list(set(all_edges_flat))
    return np.asarray(all_edges_flat_unique)


@functools.lru_cache(32)
def cached_vertex_grid_to_edges_grid_coords(grid_dim: int):
    edges_grid_idxs = edges_from_grid(grid_dim, neighbourhood_fn="4-grid")
    return edges_grid_idxs[:, 0], edges_grid_idxs[:, 1], edges_grid_idxs[:, 2], edges_grid_idxs[:, 3]


@functools.lru_cache(32)
def cached_vertex_grid_to_edges(grid_dim: int):
    x, y, xn, yn = cached_vertex_grid_to_edges_grid_coords(grid_dim)
    return np.vstack([vertex_index((x, y), grid_dim), vertex_index((xn, yn), grid_dim)]).T


def maybe_parallelize(function, arg_list):
    if ray.is_initialized():
        ray_fn = ray.remote(function)
        return ray.get([ray_fn.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]

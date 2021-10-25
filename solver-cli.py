#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

from solvers.dijkstra import get_solver


def main(argv):
    neighbourhood_fn = "8-grid"
    solver = get_solver(neighbourhood_fn)

    matrix = np.ones(shape=[8, 8], dtype=np.float)
    matrix[1:4, 0] = 0
    matrix[3, 0:3] = 0

    print(matrix)

    path = solver(matrix)

    print(path)


if __name__ == '__main__':
    main(sys.argv[1:])

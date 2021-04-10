import math

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm

from search.newton import newton


def grid_image_item(image, title=None, y_label=None, x_label=None):
    def inner(ax):
        if title is not None:
            ax.set_title(title)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if x_label is not None:
            ax.set_xlabel(x_label)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

    return inner


def default_drawer(ax, d):
    if d is None:
        plt.axis('off')
        return
    d(ax)


def axis_off():
    plt.axis('off')


def draw_grid(data, drawer=default_drawer, ncols=2, batch_size=10, hspace=.5, row_coef=16 / 2.5):
    # if len(data) > batch_size != -1:
    #     draw_grid(data[:batch_size], drawer, ncols, batch_size, hspace, row_coef)
    #     draw_grid(data[batch_size:], drawer, ncols, batch_size, hspace, row_coef)
    #     return

    nrows = math.ceil(len(data) / ncols)

    fig = plt.figure(figsize=(16, int(math.ceil(row_coef * nrows))), dpi=200)
    fig.clf()
    fig.tight_layout()
    grid = plt.GridSpec(nrows, ncols, wspace=.25, hspace=hspace)
    for i in trange(nrows * ncols):
        ax_i = i // ncols
        ax_j = i % ncols
        cur_ax = fig.add_subplot(grid[ax_i, ax_j])
        if len(data) > i:
            drawer(cur_ax, data[i])
        else:
            plt.axis('off')
    plt.show()


def draw_paths(f, path, x_min, x_max, x_step, y_min, y_max, y_step, levels, filter_mode=100):
    x_s = np.arange(x_min, x_max, x_step)
    y_s = np.arange(y_min, y_max, y_step)
    z_s = np.array([[f(np.array([x, y])) for x in x_s] for y in y_s])

    plt.contour(x_s, y_s, z_s, levels=levels)
    col = ['red', 'green', 'blue']
    algo = ['grad', 'grad_fr', 'newton']
    for k, p in enumerate(path):
        x_s = np.arange(x_min, x_max, x_step)
        y_s = np.arange(y_min, y_max, y_step)
        z_s = np.array([[f(np.array([x, y])) for x in x_s] for y in y_s])

        plt.contour(x_s, y_s, z_s, levels=levels)
        with np.printoptions(precision=8, suppress=True):
            print('Algo: ', algo[k])
            print('Result:', p[-1])
            print('Iterations:', len(p))
        points_to_show = [p[i] for i in range(len(p) - 1) if (i % filter_mode == 0) or i < 10]
        points_to_show.append(p[-1])
        for i in tqdm(range(len(points_to_show) - 1)):
            cur_point = points_to_show[i]
            next_point = points_to_show[i + 1]
            # plt.scatter([cur_point[0]], [cur_point[1]], x = col[k])
            plt.plot([cur_point[0], next_point[0]], [cur_point[1], next_point[1]], '.', c=col[k])
            plt.plot([cur_point[0], next_point[0]], [cur_point[1], next_point[1]], c=col[k])
        plt.grid()
        plt.show()

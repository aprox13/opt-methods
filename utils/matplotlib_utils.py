import math

import matplotlib.pyplot as plt
from tqdm import trange


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
    if len(data) > batch_size != -1:
        draw_grid(data[:batch_size], drawer, ncols, batch_size, hspace, row_coef)
        draw_grid(data[batch_size:], drawer, ncols, batch_size, hspace, row_coef)
        return

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

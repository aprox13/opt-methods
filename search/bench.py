# %%

from tqdm import tqdm

from core.extended_functions import MinusSin1
from one_dim_search import *
from utils.matplotlib_utils import *

# %%


RANGE = Range(0, 4)

TARGET_EPS = [1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8][::-1]

fib = Fib()
ENGINES = {
    "dichotomy": Dichotomy(find_min=True),
    "golden section": GoldenSection(find_min=True),
    "fibonacci": Fibonacci(fib)
}

# %%

RESULT = {}

for eps in tqdm(TARGET_EPS):
    RESULT[eps] = {}
    for e_name, e in ENGINES.items():
        RESULT[eps][e_name] = e.search(RANGE, MinusSin1(), eps)

engines_cnt = len(ENGINES)

fig = plt.figure()


def by_eps_plot(title: str, items: dict):
    def inner(ax):
        ax.set_title(title)
        for k, (xx, yy) in items.items():
            x_labels = list(map(str, xx))
            x_ticks = list(range(len(xx)))

            ax.set_xlabel('Eps')
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.plot(x_ticks, yy, label=k)
        ax.legend()

    return inner


def range_plt(e_name: str, eps: float):
    def _inner(ax):
        ax.set_title(f'Ranges for {e_name}[eps = {eps}]')
        res: SearchResult = RESULT[eps][e_name]

        ranges = res.ranges

        yy = list(range(res.iterations))

        xx_left = []
        xx_right = []

        for r in ranges:
            xx_left.append(r.left)
            xx_right.append(r.right)

        ax.plot(xx_left, yy)
        ax.plot(xx_right, yy)

    return _inner


iter_by_eps = {}
calls_by_eps = {}
results_by_eps = {}

for e_name in ENGINES.keys():
    xx = []
    iter_yy = []
    calls_yy = []
    results_yy = []
    for eps in TARGET_EPS:
        xx.append(eps)
        res: SearchResult = RESULT[eps][e_name]
        iter_yy.append(res.iterations)
        calls_yy.append(res.calls_count)
        results_yy.append(res.x)

    iter_by_eps[e_name] = (xx, iter_yy)
    calls_by_eps[e_name] = (xx, calls_yy)
    results_by_eps[e_name] = (xx, results_yy)

grid = [
    by_eps_plot("Iterations by eps", iter_by_eps),
    by_eps_plot("Calls by eps", calls_by_eps),
    by_eps_plot("Result by eps", results_by_eps)
]

for e_name in ENGINES.keys():
    grid.append(range_plt(e_name, 1e-4))

# %%
draw_grid(grid, ncols=engines_cnt)

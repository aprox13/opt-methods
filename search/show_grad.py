from search.grad import *

from utils.matplotlib_utils import draw_grid


def draw_grad_des(e_func: ExtendedFunction,
                  start: np.ndarray,
                  strategy: StepStrategy,
                  max_iters=30,
                  initial_step=1):
    def _drawer(ax):
        grad = GradDescent()

        trace = {
            'x': [],
            'y': [],
            'f': []
        }

        def tracer(iter_n, point: np.ndarray):
            assert point.shape == (2,), f'Expected 2, got {len(point[0])}'

            trace['x'].append(point[0])
            trace['y'].append(point[1])
            trace['f'].append(e_func(point))

        found = grad.search(
            f=e_func,
            start=start,
            step_strategy=strategy,
            max_iters=max_iters,
            initial_step=initial_step,
            before_iteration=tracer
        )

        tracer(-1, found)

        for key in trace.keys():
            trace[key] = np.array(trace[key])

        xs = trace['x']
        ys = trace['y']

        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        dx, dy = max_x - min_x, max_y - min_y
        expansion = 1
        x = np.arange(min_x - dx * expansion - 1, max_x + dx * expansion + 1, 0.01)
        y = np.arange(min_y - dy * expansion - 1, max_y + dy * expansion + 1, 0.01)
        xx, yy = np.meshgrid(x, y, sparse=True)

        z = e_func((xx, yy), unsafe=True)

        title = f'{e_func.name};{strategy.name};initial_step={initial_step}'
        ax.contour(x, y, z)
        ax.scatter(xs, ys, s=10, c="black", edgecolors="black")
        ax.plot(xs, ys, c="black")
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)

    return _drawer


def default_one_dim_sup(name):
    def _inner(f):
        return OneDimOptStrategy(f, name, search_range=Range(0, 1000), eps=1e-9)

    return _inner


INITIAL_STEPS = (0.01, 0.1, 1, 10)


def grad_shower(func: ExtendedFunction,
                start_point: np.ndarray,
                strategy_sup: Callable[[ExtendedFunction], StepStrategy],
                max_iters=15,
                initial_steps=INITIAL_STEPS
                ):
    strategy = strategy_sup(func)
    data = [draw_grad_des(func, start_point, strategy, initial_step=s, max_iters=max_iters) for s in initial_steps]

    draw_grid(data, ncols=2)

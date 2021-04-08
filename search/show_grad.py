import matplotlib.pyplot as plt

from core.extended_functions import Paraboloid
from search.grad import *

gr = GradDescent()
f = Paraboloid()
res = gr.search(f, np.array([11222, 13383]), max_iters=10, step_strategy=DividePrevStrategy(f),
                before_iteration=lambda i, point: print(f'#{i}: {point}'))


def draw_grad_des(ax, name: str, e_func: ExtendedFunction, start: np.ndarray, strategy: StepStrategy,
                  initial_step=1):
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
        max_iters=30,
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

    print(xx.shape, yy.shape)
    z = f((xx, yy), unsafe=True)

    title = f'{name}_{strategy.name}'
    print('Drawing')
    ax.contour(x, y, z)
    ax.scatter(xs, ys, s=10, c="black", edgecolors="black")
    ax.plot(xs, ys, c="black")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)


F = DelegateFunction(func=lambda x: 2 * x[0] ** 2 + x[1] ** 2, grad_func=lambda x: np.array([4 * x[0], 2 * x[1]]))

fig, ax = plt.subplots()

draw_grad_des(
    ax,
    name='p',
    e_func=Paraboloid(), start=np.array([1, 10]),
    strategy=DividePrevStrategy(f)
)
plt.show()

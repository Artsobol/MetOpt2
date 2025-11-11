import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# Преобразует строку в выражение
_ALLOWED_NAMES = {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
    'exp': np.exp, 'log': np.log, 'log10': np.log10, 'sqrt': np.sqrt, 'abs': np.abs,
    'floor': np.floor, 'ceil': np.ceil, 'round': np.round,
    'pi': np.pi, 'e': np.e,
    'sign': np.sign,
}

# Парсинг выражения
def parse_function(expr: str) -> Callable[[np.ndarray], np.ndarray]:
    expr = expr.strip()
    m = re.match(r'^\s*f\s*\(\s*x\s*\)\s*=\s*(.*)$', expr, flags=re.IGNORECASE)
    if m:
        expr = m.group(1)
    expr = expr.replace('^', '**')

    code = compile(expr, "<user_expr>", "eval")
    for name in code.co_names:
        if name not in _ALLOWED_NAMES and name != 'x':
            raise NameError(f"Недопустимое имя в выражении: {name}")

    def f(x: np.ndarray) -> np.ndarray:
        return eval(code, {"__builtins__": {}}, dict(_ALLOWED_NAMES, x=x))

    return f

# Класс для вывода результата
@dataclass
class PSResult:
    x_min: float
    f_min: float
    iters: int
    evals: int
    elapsed_sec: float
    xs: List[float]
    fs: List[float]
    L_used: float

# Глобальный поиск минимума липшицевой функции на [a,b]
class PiyavskiiShubert:
    def __init__(self, L: Optional[float] = None, r: float = 1.2):
        self.L = L
        self.r = r

    @staticmethod
    def _characteristic(xi, fi, xj, fj, L) -> float:
        # Нижняя оценка на [xi,xj]
        return 0.5*(fi + fj) - 0.5*L*(xj - xi)

    @staticmethod
    def _candidate_point(xi, fi, xj, fj, L) -> float:
        # Точка Шуберта в [xi,xj]
        x_new = 0.5*(xi + xj) - 0.5*(fj - fi)/L
        eps = 1e-12
        return float(np.clip(x_new, xi + eps, xj - eps))

    @staticmethod
    def _inflate_L(xs: List[float], fs: List[float], r: float) -> float:
        if len(xs) < 2:
            return 1.0
        order = np.argsort(xs)
        xs_s = np.array([xs[i] for i in order], dtype=float)
        fs_s = np.array([fs[i] for i in order], dtype=float)
        diffs = np.abs(np.diff(fs_s) / np.diff(xs_s))
        m = float(np.max(diffs)) if diffs.size else 1.0
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        return m * r

    def minimize(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        eps: float = 1e-3,
        max_iters: int = 2000,
    ) -> PSResult:
        t0 = time.perf_counter()
        if not (a < b):
            raise ValueError("Ожидается a < b")

        # начальные замеры
        xs: List[float] = [float(a), float(b)]
        fs: List[float] = [float(f(np.array(a))), float(f(np.array(b)))]
        evals = 2
        it = 0

        L = self.L if self.L is not None else self._inflate_L(xs, fs, self.r)
        best_idx = int(np.argmin(fs))
        x_best, f_best = xs[best_idx], fs[best_idx]

        while it < max_iters:
            it += 1
            order = np.argsort(xs)
            xs_s = [xs[i] for i in order]
            fs_s = [fs[i] for i in order]

            if self.L is None:
                L = self._inflate_L(xs, fs, self.r)

            # выбрать интервал с минимальной характеристикой
            R_min = float('inf')
            x_new = None
            left = right = None
            for i in range(len(xs_s) - 1):
                xi, fi = xs_s[i], fs_s[i]
                xj, fj = xs_s[i+1], fs_s[i+1]
                R = self._characteristic(xi, fi, xj, fj, L)
                if R < R_min:
                    R_min = R
                    x_new = self._candidate_point(xi, fi, xj, fj, L)
                    left, right = xi, xj

            if (right - left) <= eps:
                break

            f_new = float(f(np.array(x_new)))
            evals += 1
            xs.append(x_new)
            fs.append(f_new)

            if f_new < f_best:
                x_best, f_best = x_new, f_new

        elapsed = time.perf_counter() - t0
        return PSResult(
            x_min=float(x_best),
            f_min=float(f_best),
            iters=it,
            evals=evals,
            elapsed_sec=elapsed,
            xs=xs,
            fs=fs,
            L_used=float(L),
        )

# Создание графика
def plot_result(
    f: Callable[[np.ndarray], np.ndarray],
    a: float, b: float,
    result: PSResult,
    title: str = "Пиявски-Шуберт"
):
    grid = np.linspace(a, b, 1200)
    f_vals = f(grid)

    xs = np.array(result.xs)
    fs = np.array(result.fs)
    L = result.L_used

    diff = np.abs(grid[:, None] - xs[None, :])
    cones = fs[None, :] - L * diff
    lower_env = np.max(cones, axis=1)

    plt.figure(figsize=(9, 5))
    plt.plot(grid, f_vals, label="f(x)")
    plt.plot(grid, lower_env, linestyle="--", label="Нижняя огибающая")
    order = np.argsort(xs)
    plt.plot(xs[order], fs[order], marker="o", linestyle="-", label="Точки алгоритма")
    plt.scatter([result.x_min], [result.f_min], marker="x", s=100, label="Найденный минимум")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Принимает выражение, вызывает метод с основной логикой и полученный результат передает в функцию построения графика
def run_optimizer_from_string(
    expr: str, a: float, b: float, eps: float,
    L: Optional[float] = None, max_iters: int = 2000, r: float = 1.2,
    title: Optional[str] = None
) -> PSResult:
    f = parse_function(expr)
    opt = PiyavskiiShubert(L=L, r=r)
    res = opt.minimize(f, a, b, eps=eps, max_iters=max_iters)
    plot_result(f, a, b, res, title or "Пиявски-Шуберт")
    print(f"x* = {res.x_min:.6g}")
    print(f"f(x*) = {res.f_min:.6g}")
    print(f"Итераций: {res.iters}, вызовов f: {res.evals}, время: {res.elapsed_sec:.4f} с, L: {res.L_used:.6g}")
    return res

# Вызов готовых функций
def _demo(which: str):
    if which.lower() == "rastrigin":
        run_optimizer_from_string("10 + x**2 - 10*cos(2*pi*x)", -4.0, 4.0, eps=5e-3, L=None, title="График")
    elif which.lower() == "ackley":
        run_optimizer_from_string("-20*exp(-0.2*abs(x)) - exp(cos(2*pi*x)) + 20 + e", -5.0, 5.0, eps=5e-3, L=None, title="График")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Глобальный минимум на [a,b] методом Пиявского–Шуберта")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--expr", type=str, help="Строка выражения, например: \"f(x)=x+sin(pi*x)\" или \"10 + x**2\"")
    g.add_argument("--demo", type=str, choices=["rastrigin", "ackley"], help="Запуск одного из демо")

    p.add_argument("--a", type=float, default=-4.0, help="Левая граница отрезка")
    p.add_argument("--b", type=float, default=4.0, help="Правая граница отрезка")
    p.add_argument("--eps", type=float, default=1e-3, help="Требуемая точность по x (критерий остановки по ширине активного интервала)")
    p.add_argument("--L", type=float, default=None, help="Известная константа Липшица. Если не задана, оценивается адаптивно")
    p.add_argument("--r", type=float, default=1.2, help="Коэффициент запаса для L при адаптивной оценке")
    p.add_argument("--max-iters", type=int, default=2000, help="Ограничение числа итераций")
    p.add_argument("--title", type=str, default=None, help="Заголовок графика")

    args = p.parse_args()

    if args.demo:
        _demo(args.demo)
    else:
        run_optimizer_from_string(args.expr, args.a, args.b, args.eps, L=args.L, r=args.r, max_iters=args.max_iters, title=args.title)

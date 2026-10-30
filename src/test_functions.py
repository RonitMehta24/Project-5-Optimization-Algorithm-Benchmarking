
import numpy as np


class SphereFunction:
    """Sphere function: f(x, y) = x^2 + y^2."""

    TOLERANCE = 1e-6
    MAX_ITERATIONS = 10000

    @staticmethod
    def objective(x):
        return x[0]**2 + x[1]**2

    @staticmethod
    def gradient(x):
        return np.array([
            2*x[0],
            2*x[1]
        ])

    @staticmethod
    def hessian(x):
        return np.array([
            [2.0, 0.0],
            [0.0, 2.0]
        ])

    @staticmethod
    def solution():
        return np.array([[0.0, 0.0]])

    @staticmethod
    def minimum_value():
        return 0.0

    @staticmethod
    def stopping_criterion(x, iteration):
        return (
            np.linalg.norm(SphereFunction.gradient(x))
            <= SphereFunction.TOLERANCE
            or iteration >= SphereFunction.MAX_ITERATIONS
        )


class BoothFunction:
    """Booth function with minimum at (1, 3)."""

    TOLERANCE = 1e-6
    MAX_ITERATIONS = 10000

    @staticmethod
    def objective(x):
        a = x[0] + 2*x[1] - 7
        b = 2*x[0] + x[1] - 5
        return a**2 + b**2

    @staticmethod
    def gradient(x):
        a = x[0] + 2*x[1] - 7
        b = 2*x[0] + x[1] - 5
        return np.array([
            2*a + 4*b,
            4*a + 2*b
        ])

    @staticmethod
    def hessian(x):
        return np.array([
            [10.0, 8.0],
            [8.0, 10.0]
        ])

    @staticmethod
    def solution():
        return np.array([[1.0, 3.0]])

    @staticmethod
    def minimum_value():
        return 0.0

    @staticmethod
    def stopping_criterion(x, iteration):
        return (
            np.linalg.norm(BoothFunction.gradient(x))
            <= BoothFunction.TOLERANCE
            or iteration >= BoothFunction.MAX_ITERATIONS
        )


class RosenbrockFunction:
    """Rosenbrock function with minimum at (1, 1)."""

    TOLERANCE = 1e-6
    MAX_ITERATIONS = 10000

    @staticmethod
    def objective(x):
        return (
            100*(x[1] - x[0]**2)**2
            + (1 - x[0])**2
        )

    @staticmethod
    def gradient(x):
        a = x[1] - x[0]**2
        return np.array([
            -400*x[0]*a - 2*(1 - x[0]),
            200*a
        ])

    @staticmethod
    def hessian(x):
        return np.array([
            [1200*x[0]**2 - 400*x[1] + 2,
             -400*x[0]],
            [-400*x[0], 200.0]
        ])

    @staticmethod
    def solution():
        return np.array([[1.0, 1.0]])

    @staticmethod
    def minimum_value():
        return 0.0

    @staticmethod
    def stopping_criterion(x, iteration):
        return (
            np.linalg.norm(RosenbrockFunction.gradient(x))
            <= RosenbrockFunction.TOLERANCE
            or iteration >= RosenbrockFunction.MAX_ITERATIONS
        )


class HimmelblauFunction:
    """Himmelblau function with four global minimizers."""

    TOLERANCE = 1e-6
    MAX_ITERATIONS = 10000

    @staticmethod
    def objective(x):
        a = x[0]**2 + x[1] - 11
        b = x[0] + x[1]**2 - 7
        return a**2 + b**2

    @staticmethod
    def gradient(x):
        a = x[0]**2 + x[1] - 11
        b = x[0] + x[1]**2 - 7
        return np.array([
            4*x[0]*a + 2*b,
            2*a + 4*x[1]*b
        ])

    @staticmethod
    def hessian(x):
        a = x[0]**2 + x[1] - 11
        b = x[0] + x[1]**2 - 7
        return np.array([
            [4*a + 8*x[0]**2 + 2,
             4*x[0] + 4*x[1]],
            [4*x[0] + 4*x[1],
             2 + 4*b + 8*x[1]**2]
        ])

    @staticmethod
    def solution():
        return np.array([
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126]
        ])

    @staticmethod
    def minimum_value():
        return 0.0

    @staticmethod
    def stopping_criterion(x, iteration):
        return (
            np.linalg.norm(HimmelblauFunction.gradient(x))
            <= HimmelblauFunction.TOLERANCE
            or iteration >= HimmelblauFunction.MAX_ITERATIONS
        )

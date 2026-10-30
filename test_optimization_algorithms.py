
import numpy as np

from src.algorithms import Optimization
from src.test_functions import SphereFunction


# Test Newton's Method
def test_sphere_newton_method():

    optimizer = Optimization(
        SphereFunction.objective,
        SphereFunction.gradient,
        SphereFunction.hessian
    )

    initial_point = np.array([5.0, 5.0])

    final_point, history, iterations = optimizer.newton_method(
        initial_point
    )

    assert np.allclose(final_point, [0.0, 0.0], atol=1e-6)


# Test Steepest Descent
def test_sphere_steepest_descent():

    optimizer = Optimization(
        SphereFunction.objective,
        SphereFunction.gradient,
        SphereFunction.hessian
    )

    initial_point = np.array([5.0, 5.0])

    final_point, history, iterations = optimizer.steepest_descent(
        initial_point
    )

    assert np.allclose(final_point, [0.0, 0.0], atol=1e-4)

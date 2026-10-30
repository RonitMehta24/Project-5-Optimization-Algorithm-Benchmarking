
import numpy as np

from src.test_functions import (
    SphereFunction,
    BoothFunction,
    RosenbrockFunction,
    HimmelblauFunction,
)


FUNCTIONS = [
    SphereFunction,
    BoothFunction,
    RosenbrockFunction,
    HimmelblauFunction,
]


def test_function_class(function_class):
    solutions = function_class.solution()

    for x in solutions:
        assert np.isclose(
            function_class.objective(x),
            function_class.minimum_value(),
            atol=1e-7
        )

        assert np.linalg.norm(
            function_class.gradient(x)
        ) < 1e-4

        hessian = function_class.hessian(x)

        assert hessian.shape == (2, 2)
        assert np.allclose(hessian, hessian.T)

    print(function_class.__name__, "passed")


if __name__ == "__main__":
    for function_class in FUNCTIONS:
        test_function_class(function_class)

    print("All function tests passed.")

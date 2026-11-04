# Optimization Algorithms (`algorithms.py`)

## Overview
The `algorithms.py` module houses the `Optimization` class, which provides implementations for unconstrained numerical optimization. The class is designed to be initialized with any objective function, decoupling the optimization logic from the mathematical test cases.

## Class: `Optimization`

### Initialization
```python
Optimization(objective, gradient, hessian)
```
*   **`objective`**: Callable function returning the scalar objective value.
*   **`gradient`**: Callable function returning the gradient array.
*   **`hessian`**: Callable function returning the 2D Hessian matrix.

### Core Methods

#### 1. `steepest_descent(x0, tolerance=1e-6, step_size=0.1, maximum_iterations=100)`
Executes the steepest descent algorithm using the negative gradient direction with a fixed step size.
*   **Inputs:**
    *   `x0`: Initial starting array.
    *   `tolerance`: Threshold for the gradient norm to declare convergence.
    *   `step_size`: Fixed scalar multiplier for the gradient step.
    *   `maximum_iterations`: Hard limit on the number of steps.
*   **Outputs:** Returns a tuple `(final_x, history, converged, iterations)`.

#### 2. `newton_method(x0, tol=1e-6, max_iter=100, fallback_step_size=1e-3, condition_threshold=1e12)`
Executes Newton's method by solving the linear system $H d = -g$ using the exact Hessian.
*   **Safety Checks:** Validates that the Hessian is symmetric, finite, and positive definite via Cholesky decomposition. It checks the condition number against `condition_threshold`.
*   **Fallback Mechanism:** If the Hessian is singular, non-positive definite, or ill-conditioned, it catches the `np.linalg.LinAlgError` and defaults to a gradient descent step using the `fallback_step_size`.
*   **Outputs:** Returns a tuple `(final_x, history, converged, iterations)`.

#### 3. `evaluate_and_record_history(x, history)`
A diagnostic helper method that records the current state of the optimizer.
*   **Functionality:** Appends copies of the current iterate, objective value, gradient array, and gradient norm to the `history` dictionary. 
*   **Safety Checks:** Raises `ValueError` if non-finite values are detected in the objective or gradient, preventing silent mathematical failures.
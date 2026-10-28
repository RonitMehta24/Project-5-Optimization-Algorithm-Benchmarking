# Design of Optimization Test Function Classes

## 1. Objective

The purpose of these classes is to provide standard test functions for evaluating numerical optimization algorithms. Each function is implemented as an independent class with a common interface.

## 2. Common Class Structure

Every class contains the following components:

- `objective(x)`: Static method that evaluates the objective function.
- `gradient(x)`: Static method that computes the gradient vector.
- `hessian(x)`: Static method that computes the Hessian matrix.
- `solution()`: Static method that returns the known global minimizer or minimizers.
- `minimum_value()`: Static method that returns the known minimum objective value.
- `stopping_criterion(x, iteration)`: Static method that determines whether the stopping condition has been met.

Each class also defines a gradient tolerance and a maximum iteration count.

## 3. Selected Test Functions

### Sphere Function

Objective: f(x, y) = x² + y²

Global minimizer: (0, 0)

Minimum value: 0

### Booth Function

Objective: f(x, y) = (x + 2y - 7)² + (2x + y - 5)²

Global minimizer: (1, 3)

Minimum value: 0

### Rosenbrock Function

Objective: f(x, y) = 100(y - x²)² + (1 - x)²

Global minimizer: (1, 1)

Minimum value: 0

### Himmelblau Function

Objective: f(x, y) = (x² + y - 11)² + (x + y² - 7)²

Global minimizers:
(3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), and (3.584428, -1.848126).

Minimum value: 0.

## 4. Stopping Criterion

The implementation uses the following example stopping rule:

Stop when the Euclidean norm of the gradient is less than or equal to the specified tolerance, or when the maximum number of iterations is reached.

Tolerance: 10⁻⁶

Maximum iterations: 10,000

The stopping rule may be adjusted to match the optimization algorithm or the project specification.

## 5. Design Benefits

The consistent interface allows an optimization algorithm to access any test function using the same method names. Separate classes make the code modular, reusable, and easy to extend.

## 6. Validation

The objective functions, gradients, Hessians, analytic solutions, and stopping criteria should be validated using independent numerical tests before the classes are used in optimization experiments.

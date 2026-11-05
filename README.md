# Project-5-Optimization-Algorithm-Benchmarking

## Overview
This repository implements and compares unconstrained numerical optimization methods—specifically Steepest Descent and Newton's Method. It benchmarks these algorithms against standard mathematical test functions (Sphere, Booth, Rosenbrock, and Himmelblau) to investigate the effects of step size, local curvature, and initial starting points on convergence behavior.

## Project Structure
* `src/algorithms.py`: Core implementations of Steepest Descent and Newton's Method.
* `src/test_functions.py`: Mathematical objective functions, including analytical gradients and Hessians.
* `src/benchmarking.py`: Interactive CLI script for running experiments and generating convergence plots.
* `tests/`: Unit tests verifying the mathematical correctness of the objective functions and their derivatives.
* `docs/`: Module design documentation and architecture plans.
* `report/`: LaTeX source file and pdf file for the final project report.

## Requirements
* Python 3
* NumPy
* Matplotlib

## Usage
Run the benchmarking file interactively from the root directory:

```bash
python src/benchmarking.py
```

The script will prompt you through a menu to select:
1. **The Test Function:** Choose between Sphere, Booth, Rosenbrock, or Himmelblau.
2. **The Algorithm:** Run Steepest Descent, Newton's Method, or compare both side-by-side.
3. **Parameters:** Specify a custom starting coordinate point and, if applicable, the fixed step size for steepest descent.
4. **Visualization:** Choose whether to display and export contour plots and objective value trajectories.
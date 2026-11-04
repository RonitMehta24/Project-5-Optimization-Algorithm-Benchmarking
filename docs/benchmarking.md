# Benchmarking and visualization

The 'benchmarking.py' file runs steepest descent and Newton's method on test functions. It prints the results and plots how the methods behave. The optimization algorithms are in 'algorithms.py', and the test functions are in 'test_functions.py'.

## Creating a benchmark

The 'Benchmark' class takes a test function, an optimization class, an initial point, and an optional step size.

benchmark = Benchmark(
    test_class=SphereFunction,
    optimization_class=algorithms.Optimization,
    x0=[5.0, 5.0],
    step_size=0.1
)

Here, SphereFunction is the function being minimized, [5.0, 5.0] is the starting point, and 0.1 is the fixed step size for steepest descent. The default step size is 0.001. This argument is not passed to Newton's method.

The available test functions are SphereFunction, BoothFunction, RosenbrockFunction, and HimmelblauFunction.

The tolerance and maximum iteration limit come from the test function class. Each run starts from a copy of the initial point so that both methods can be compared from the same starting position.

## Running steepest descent

benchmark.run_steepest_descent()

This runs steepest descent and prints the function name, starting point, step size, final point, final objective value, iteration count, convergence status, final gradient norm, and distance from the nearest known minimizer. It also shows a plot of steepest descent.

To print the results without plotting:

benchmark.run_steepest_descent(plot=False)

## Running Newton's method

benchmark.run_newton_method()

This runs Newton's method and prints its results. It shows the same information as steepest descent except for the fixed step size. Its plot shows only Newton's method.

To print the results without plotting:

benchmark.run_newton_method(plot=False)


## Comparing both methods

benchmark.compare_methods()

This clears the previous results and runs both methods from the same initial point. Each method prints its own results once. The individual plots are disabled during these runs, and one combined figure is shown afterward.

In this version, compare_methods() always shows the combined plot and does not take a plot argument. To get both sets of results without plotting, call the two run methods separately with plot=False.

## Measuring execution time

Both run methods use `perf_counter()` from Python's `time` module to measure elapsed execution time in seconds.

The timer starts immediately before the optimizer call and stops immediately after it returns. The measured interval includes preparing the copied starting point, optimization updates, convergence checks, and iteration history recording. Printing, storing the result summary, and plotting happen after the timer stops.

Each optimizer runs once. Timing is written directly inside `run_newton_method()` and `run_steepest_descent()`; no separate timing helper is needed.

## Results and plots

Both run methods return the final point, iteration history, convergence status, and iteration count:

x, history, converged, iteration_count = benchmark.run_steepest_descent(plot=False)

Results are also stored in 'benchmark.results', under either "Steepest Descent" or "Newton's Method". Each entry contains the final point, history, convergence status, iteration count, final objective value, and distance from the nearest known minimizer.

result = benchmark.results["Steepest Descent"]
print(result["final_objective"])

The history records the points, objective values, gradients, and gradient norms. Running the same method again replaces its previous result.

Each figure has two panels. The left panel shows objective value against iteration. The right panel shows the optimization path over a contour plot of the objective function. The starting point is a yellow square, final points are marked with an X, and known minima are red stars.

When comparing methods, both Newton and steepest descent appear in each panel with separate legend labels. The contour plot is designed for two-dimensional functions.

You can also plot results that have already been stored:

benchmark.plot_history("Steepest Descent")
benchmark.plot_history("Newton's Method")
benchmark.plot_history()

The last call plots all stored methods. A method must have been run before its results can be plotted.

## Running experiments

Run the program from the repository folder:

"python src/benchmarking.py"

The 'if __name__ == "__main__":' block uses 'input()' to collect experiment settings from the terminal. Importing the file from another Python file does not start the input menu.

First, choose the function: '1' for Sphere, '2' for Booth, '3' for Rosenbrock, or '4' for Himmelblau. A dictionary maps the choice to its function class, default starting point, and default steepest-descent step size. Users choose an existing function rather than entering a mathematical expression.

Next, choose the experiment: '1' for steepest descent, '2' for Newton's method, or '3' to compare both.

Enter the starting point as two numbers separated by a space, such as '5 5' or '-1.2 1'. Do not use brackets or commas. The program splits the text and converts it into a floating-point NumPy array. It checks that there are exactly two coordinates and both are finite. Press Enter to use the displayed default.

The step-size prompt appears only for steepest descent or a comparison. The input is converted using `float()` and must be positive and finite. Press Enter to use the default. Newton's method does not receive this fixed step size.

The defaults are '[5, 5]' and step '0.1' for Sphere, '[0, 0]' and step '0.01' for Booth, '[-1.2, 1]' and step '0.001' for Rosenbrock, and `[0, 0]' and step '0.001' for Himmelblau.

Finally, enter `y` to show plots or `n` for printed results only. Press Enter to show plots. The program creates a 'Benchmark' object from the selected settings and calls the chosen run method or 'compare_methods(plot=plot)'.

For example, this input sequence compares both methods on Sphere:

"Function number: 1
Experiment number: 3
Initial point: 5 5
Steepest-descent step size: 0.1
Show plots? y"

Both methods print their numerical results and execution times. If plots were requested, their curves and trajectories appear in one combined figure.

Invalid selections, malformed coordinates, and invalid step sizes produce an error message and end the run. The 'try except' displays handled errors and ends the experiment; cancelling input also ends it. The program does not automatically retry. Run the command again for another experiment.
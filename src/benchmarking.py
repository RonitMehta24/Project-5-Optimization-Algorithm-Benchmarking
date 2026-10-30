import algorithms
from matplotlib.colors import LogNorm

from test_functions import (
    SphereFunction,
    BoothFunction,
    RosenbrockFunction,
    HimmelblauFunction
)

import numpy as np
import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self, test_class, optimization_class, x0, step_size = 0.001):
        
        #Extracting information from test class
        self.test_class = test_class
        self.objective = test_class.objective
        self.gradient = test_class.gradient
        self.hessian = test_class.hessian

        #Store the known global minimizers and minimum objective value
        self.solution = np.asarray(test_class.solution(), dtype = float)
        self.minimum_value = test_class.minimum_value()

        #Get the stopping tolerance and maximum iterations defined in the test function class
        self.tolerance = test_class.TOLERANCE
        self.max_iter = test_class.MAX_ITERATIONS
        
        #Store the initial point and step size
        #copy() prevents accidental modification of the original x0
        self.x0 = np.asarray(x0, dtype = float).copy()
        self.step_size = step_size


        #Initialize the optimization object using the selected
        #objective function, gradient and Hessian
        self.optimizer = optimization_class(self.objective, self.gradient, self.hessian)

        self.results = {}  #Dictionary to store results

    def run_newton_method(self, plot = True):

        x , history, converged, iteration_count = (
                        self.optimizer.newton_method(x0 = self.x0.copy(),
                        tol = self.tolerance, 
                        max_iter = self.max_iter))
        self.results["Newton's Method"] = {
            "x": np.asarray(x, dtype = float),
            "history": history,
            "converged": converged,
            "iteration_count": iteration_count,
            "final_objective":float(self.objective(x)),
            "distance_from_solution": self.distance_from_solution(x) 
        }

        if plot:    # Plot the optimization history if requested
            self.plot_history("Newton's Method")
        return x, history, converged, iteration_count

    def run_steepest_descent(self, plot = True):
            
            x , history, converged, iteration_count = (
                            self.optimizer.steepest_descent(x0 = self.x0.copy(),
                            tolerance = self.tolerance,
                            step_size=self.step_size, 
                            maximum_iterations = self.max_iter))
            self.results["Steepest Descent"] = {
                "x": np.asarray(x, dtype = float),
                "history": history,
                "converged": converged,
                "iteration_count": iteration_count,
                "final_objective": float(self.objective(x)),
                "distance_from_solution": self.distance_from_solution(x)
            }
            if plot:
                self.plot_history("Steepest Descent")
            return x, history, converged, iteration_count
    
    
    
    def compare_methods(self):

        #Clear previous results before starting a new comparison
        self.results = {}

        # Run both methods
        self.run_newton_method(plot=False)
        self.run_steepest_descent(plot=False)

        # Display comparison
        print("\nComparison of Optimization Methods")
        print("Test Function:", self.test_class.__name__)

        for name, result in self.results.items():

            print("\n", name)
            print("Final point:", result["x"])
            print("Final objective:", result["final_objective"])
            print("Iterations:", result["iteration_count"])
            print("Converged:", result["converged"])

            gradient_norm = result["history"]["gradient_norm"][-1]
            print("Final gradient norm:", gradient_norm)

            print(
                "Distance from solution:",
                self.distance_from_solution(result["x"])
            )

        # Plot both methods together
        self.plot_history()

        return self.results

        

    def distance_from_solution(self, x):
        distances = np.linalg.norm(self.solution - x, axis=1)
        return np.min(distances)


    def plot_history(self, method = None):
        
        #Check whether optimization results are available
        if not self.results:
            print("Run an optimization method first")
            return


        
        if method is None:
            selected_results = self.results
        else:
            if method not in self.results:
                print(f"Method'{method}' has not been run.")
                return
            selected_results = {method: self.results[method]}

        fig, axes = plt.subplots(1, 2, figsize = (14,5))

        for name, result in selected_results.items():
            values = result["history"]["objective_value"]

            axes[0].plot(range(len(values)), values, marker = "o", markersize = 3, label = name)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Objective value")
        axes[0].set_title("Objective value vs iteration")
        axes[0].legend()
        axes[0].grid(True)

        all_points = [self.x0]

        for result in selected_results.values():
            all_points.extend(result["history"]["x"])

        all_points = np.asarray(all_points)

        # Include known minimizers when choosing plot limits
        plot_points = np.vstack((all_points, self.solution))

        lower = np.min(plot_points, axis=0) - 1.0
        upper = np.max(plot_points, axis=0) + 1.0

        x1 = np.linspace(lower[0], upper[0], 150)
        x2 = np.linspace(lower[1], upper[1], 150)

        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                point = np.array([X1[i, j], X2[i, j]])
                Z[i, j] = self.objective(point)

        contour = axes[1].contourf(
            X1, X2, Z,
            levels=30,
            cmap="viridis"
        )
        # Add contour lines
        axes[1].contour(
            X1, X2, Z,
            levels=30,
            colors="black",
            linewidths=0.4,
            alpha=0.4
        )

        fig.colorbar(contour, ax=axes[1], label="Objective value")

        for name, result in selected_results.items():
            path = np.array(result["history"]["x"])

            axes[1].plot(
                path[:, 0],
                path[:, 1],
                marker="o",
                markersize=3,
                markevery=max(1, len(path) // 15),
                linewidth=1.5,
                label=name
            )

            # Mark the final point
            axes[1].scatter(
                path[-1, 0],
                path[-1, 1],
                marker="X",
                s=100,
                edgecolors="black",
                label=name + " final point"
            )

            # Add arrows to indicate direction of movement
            for i in range(0, len(path) - 1, max(1, len(path) // 10)):

                axes[1].annotate(
                    "",
                    xy=path[i + 1],
                    xytext=path[i],
                    arrowprops=dict(arrowstyle="->", color="white")
                )

        # Mark initial point
        axes[1].scatter(
            self.x0[0],
            self.x0[1],
            marker="s",
            s=100,
            color="white",
            edgecolors="black",
            label="Initial point",
            zorder=5
        )

        # Mark known minimizers
        axes[1].scatter(
            self.solution[:, 0],
            self.solution[:, 1],
            marker="*",
            s=180,
            color="red",
            edgecolors="black",
            label="Known minima",
            zorder=6
        )

        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")
        axes[1].set_title("Optimization Trajectories")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        plt.show()



# for testing benchmark implementation
if __name__ == "__main__":

    test_cases = [
        (SphereFunction, np.array([5.0, 5.0]), 0.1),
        (BoothFunction, np.array([0.0, 0.0]), 0.01),
        (RosenbrockFunction, np.array([-1.2, 1.0]), 0.001),
        (HimmelblauFunction, np.array([0.0, 0.0]), 0.001)
    ]

    for function, initial_point, step_size in test_cases:

        benchmark = Benchmark(
            test_class=function,
            optimization_class=algorithms.Optimization,
            x0=initial_point,
            step_size=step_size
        )

        benchmark.compare_methods()

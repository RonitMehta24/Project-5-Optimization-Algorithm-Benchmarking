import algorithms
#import tests

import numpy as np
import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self, test_class, optimization_class):
        
        #Extracting information from test class
        self.objective = test_class.objective
        self.gradient = test_class.gradient
        self.hessian = test_class.hessian

        #initializing these parameters manually now, will be updated later
        self.solution = np.array([0.0, 0.0])
        self.tolerance = 1e-6
        self.x0 = np.array([5.0, 5.0])
        self.step_size = 0.1
        self.max_iter = 200

        self.optimizer = optimization_class(self.objective, self.gradient, self.hessian)

        self.results = {}

    def run_newton_method(self, plot = True):
        x , history, iteration_count = (
                        self.optimizer.newton_method(x0 = self.x0.copy(),
                        tol = self.tolerance, 
                        max_iter = self.max_iter))
        self.results["Newton's Method"] = {
            "x": np.asarray(x, dtype = float),
            "history": history,
            "iteration_count": iteration_count
        }
        if plot:
            self.plot_history("Newton's Method")
        return x, history, iteration_count

    def run_steepest_descent(self, plot = True):
            x , history, iteration_count = (
                            self.optimizer.steepest_descent(x0 = self.x0.copy(),
                            tol = self.tolerance, 
                            max_iter = self.max_iter))
            self.results["Newton's Method"] = {
                "x": np.asarray(x, dtype = float),
                "history": history,
                "iteration_count": iteration_count
            }
            if plot:
                self.plot_history("Newton's Method")
            return x, history, iteration_count
    
    #have to add a compare function
    def compare_methods(self):
        pass


    def plot_history(self, method = None):
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
            values = result["history"]["objective"]

            axes[0].plot(range(len(values)), values, marker = "o", markersize = 3, label = name)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Objective value")
        axes[0].set_title("Objective value vs iteration")
        axes[0].legend()
        axes[0].grid(True)

        x1 = np.linspace(-10, 100, 100)
        x2 = np.linspace(-10, 110, 100)
        X1, X2 = np.meshgrid(x1, x2)

        Z = np.zeros_like(X1)

        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i, j] = self.objective(np.array([X1[i,j], X2[i,j]]))
        axes[1].contourf(X1, X2, Z, levels = 20, cmap = "viridis")
        for name, result in selected_results.items():
            path = np.asarray(result["history"]["x"])
            axes[1].plot(path[:,0], path[:,1], "o", label = name)

        axes[1].plot(self.solution[0], self.solution[1], "r*", markersize = 12, label = "Minimum")
        axes[1].set_xlabel("$x_1$")
        axes[1].set_ylabel("$x_2$")
        axes[1].set_title("Optimization path")
        axes[1].legend()

        plt.show()


# Temporary test function
class QuadraticTest:
    @staticmethod
    def objective(x):
        return x[0]**2 + x[1]**2

    @staticmethod
    def gradient(x):
        return np.array([2*x[0], 2*x[1]])

    @staticmethod
    def hessian(x):
        return np.array([
            [2, 0],
            [0, 2]
        ])

benchmark = Benchmark(test_class = QuadraticTest, optimization_class = algorithms.Optimization)
x, history, iteration_count = (benchmark.run_newton_method(plot = False))

print("x:", x)
print("Ieration count:", iteration_count)
print("Objective history:", history["objective"])
print("Converged:", history["converged"])
benchmark.plot_history("Newton's Method")
import numpy as np

class Optimization:
    def __init__(self, objective, gradient, hessian):
        self.objective = objective 
        self.gradient = gradient
        self.hessian = hessian
        return

    # helper method for recording history
    # The dictionary "history" should have the keys mentioned below, already having a python array as its corresponding value(empty or otherwise) 
    def evaluate_and_record_history(self, x, history):
        # calculate the values necessary for iteration
        objective_value = float(self.objective(x))
        gradient_value = np.asarray(self.gradient(x), dtype = float)
        gradient_norm = np.linalg.norm(gradient_value)

        # save these values in the history of the iteration
        history["x"].append(x.copy())
        history["objective_value"].append(objective_value)
        history["gradient_value"].append(gradient_value.copy())
        history["gradient_norm"].append(gradient_norm)
        return objective_value, gradient_value, gradient_norm

    def steepest_descent(self, x0, step_size = 1e-1, tolerance = 1e-6, maximum_iterations = 100):
        x = np.asarray(x0, dtype = float).copy()
        history = {"x": [], "objective_value": [], "gradient_value": [], "gradient_norm": []}
        converged = False        
        
        for number_of_iterations in range(maximum_iterations):
            objective_value, gradient_value, gradient_norm = self.evaluate_and_record_history(x, history)

            # check if stopping criterion is met
            if gradient_norm < tolerance:
                converged = True
                return x, history, converged, number_of_iterations

            # move to the next iteration if it is not
            x -= step_size*gradient_value

        # if we exited out of the loop it means we iterated the maximum number of times
        # so now we need to add the information related to the last x value in history
        objective_value, gradient_value, gradient_norm = self.evaluate_and_record_history(x, history)
        
        # and we need to check if this x value falls within the tolerance for the convergence check
        if gradient_norm < tolerance:
            converged = True

        return x, history, converged, maximum_iterations


    #Evaluate the objective and gradient, and record the iteration.
    def evaluate_and_record_history(self, x, history):
        objective_value = float(self.objective(x))
        gradient_value = np.asarray(self.gradient(x), dtype=float)

        if gradient_value.shape != x.shape:
            raise ValueError("Gradient shape must match the shape of x.")

        if not np.isfinite(objective_value):
            raise ValueError("Objective returned a non-finite value.")

        if not np.all(np.isfinite(gradient_value)):
            raise ValueError("Gradient contains non-finite values.")

        gradient_norm = float(np.linalg.norm(gradient_value))

        history["x"].append(x.copy())
        history["objective_value"].append(objective_value)
        history["gradient_value"].append(gradient_value.copy())
        history["gradient_norm"].append(gradient_norm)

        return objective_value, gradient_value, gradient_norm

    #Basic functionality for newton's method withoput any critical case handling 
    def newton_method(self,x0,tol=1e-6,max_iter=100,fallback_step_size=1e-3,condition_threshold=1e12,):
        x = np.asarray(x0, dtype=float).copy()

        if x.ndim != 1 or x.size == 0:
            raise ValueError("x0 must be a non-empty one-dimensional array.")

        if not np.all(np.isfinite(x)):
            raise ValueError("x0 contains non-finite values.")

        if tol <= 0 or not np.isfinite(tol):
            raise ValueError("tol must be a positive finite number.")

        if max_iter < 0 or not isinstance(max_iter, (int, np.integer)):
            raise ValueError("max_iter must be a non-negative integer.")

        if fallback_step_size <= 0 or not np.isfinite(fallback_step_size):
            raise ValueError("fallback_step_size must be positive and finite.")

        if condition_threshold <= 1 or not np.isfinite(condition_threshold):
            raise ValueError("condition_threshold must be finite and greater than 1.")

        # Store the information for every evaluated iterate.
        history = { "x": [],"objective_value": [],"gradient_value": [],"gradient_norm": []}

        for k in range(max_iter + 1):
            objective_value, gradient_value, gradient_norm = self.evaluate_and_record_history(x, history)

            # Stop if the gradient norm is below the specified tolerance.
            if gradient_norm < tol:
                return x, history, True, k
            # Stop if the maximum number of iterations is reached.
            if k == max_iter:
                break

            H = np.asarray(self.hessian(x), dtype=float)

            # Validatimg the Hessian
            if H.shape != (x.size, x.size):
                raise ValueError("Hessian must be a square matrix matching the dimension of x.")
            if not np.all(np.isfinite(H)):
                raise ValueError("Hessian is containing non-finite values.")
            if not np.allclose(H, H.T, rtol=1e-7, atol=1e-10): #Symmetric check for Hessian
                raise ValueError(
                    f"Hessian is not symmetric at iteration {k}. "
                    "Check the Hessian function definition."
                )

            # Attempting a Newton step only when the Hessian is sufficiently well-conditioned and positive definite.
            try:
                condition_number = np.linalg.cond(H)

                if (
                    not np.isfinite(condition_number)
                    or condition_number > condition_threshold
                ):
                    raise np.linalg.LinAlgError("Hessian is singular or nearly singular.")

                # Check for positive definiteness.
                np.linalg.cholesky(H)

                # Solve H d = -g without explicitly calculating H inverse.
                d = np.linalg.solve(H, -gradient_value)

                if not np.all(np.isfinite(d)):
                    raise np.linalg.LinAlgError("Newton direction contains non-finite values.")
                x = x + d
                if not np.all(np.isfinite(x)):
                    raise FloatingPointError("Newton step produced non-finite values.")

            except np.linalg.LinAlgError:
                #incase of singular or nearly singular Hessian, fallback to a gradient descent step.
                x = x - fallback_step_size * gradient_value

                if not np.all(np.isfinite(x)):
                    raise FloatingPointError("Fallback step produced non-finite values.")

        # Maximum number of updates reached without convergence.
        return x, history, False, max_iter


#Basic testing function
def f(x):
    return x[0]*x[0] + x[1]*x[1]

def g(x):
    return np.array([2*x[0], 2*x[1]])

def h(x):
    return np.array([[2, 0], [0, 2]])

test = Optimization(f, g, h)
x, y, z = test.newton_method(np.array([100, 100]))
print("Newton Method test:-")
print("The final x vaule: ", x)
print("Number of iterations stored in histoy: ", len(y["x"]))
print("Number of iterations: ", z)

#testing steepest descent when convergence occurs before reaching maximum iterations
x, y, z, w = test.steepest_descent(np.array([10, 10]), 0.2)
print("Steepest Descent test(not reaching maximum iterations):-")
print("The final x value:", x)
print("Number of iterations stored in history: ", len(y["x"]))
print("Convergence: ", z)
print("Number of iterations: ", w)

#testing steepest descent when convergence does not occur before reaching maximum iterations
x, y, z, w = test.steepest_descent(np.array([100, 100]), 0.01)
print("Steepest Descent test(reaching maximum iterations):-")
print("The final x value:", x)
print("Number of iterations stored in history: ", len(y["x"]))
print("Convergence: ", z)
print("Number of iterations: ", w)

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
        # so now it is enough to check if the current x value meets the stopping criterion
        gradient_value = np.asarray(self.gradient(x), dtype = float)
        gradient_norm = np.linalg.norm(gradient_value)
        
        if gradient_norm < tolerance:
            converged = True

        return x, history, converged, number_of_iterations

        

    #Basic functionality for newton's method without any critical case handling
    def newton_method(self,x0,tol=1e-6,max_iter=100,fallback_step_size=1e-3):
        x = np.asarray(x0, dtype=float).copy()

        history = {"x": [],"objective": [],"gradient_norm": [],"converged": False,}

        def record_iteration(x, g):
            f_x = float(self.objective(x))

            history["x"].append(x.copy())
            history["objective"].append(f_x)
            history["gradient_norm"].append(float(np.linalg.norm(g)))

        for k in range(max_iter + 1):
            g = np.asarray(self.gradient(x), dtype=float)

            record_iteration(x, g)

            if np.linalg.norm(g) < tol:
                history["converged"] = True
                return x, history, k

            if k == max_iter:
                break

            H = np.asarray(self.hessian(x), dtype=float)

            # Newton's direction needs a suitable Hessian.
            # Use steepest descent if the Hessian is not positive definite.
            try:
                if not np.allclose(H, H.T):
                    raise ValueError("Hessian is not symmetric.")

                np.linalg.cholesky(H)
                d = np.linalg.solve(H, -g) #Solving Hd = -g for d
                x = x + d

            except (ValueError, np.linalg.LinAlgError):
                # Fallback to a small steepest-descent step.
                x = x - fallback_step_size * g

        return x, history, max_iter

#Basic testing function
def f(x):
    return x[0]*x[0] + x[1]*x[1]

def g(x):
    return np.array([2*x[0], 2*x[1]])

def h(x):
    return np.array([[2, 0], [0, 2]])

test = Optimization(f, g, h)
x, y, z = test.newton_method(np.array([100, 100]))
print(x)
#print(y)
print(z)
x, y, z, w = test.steepest_descent(np.array([10, 10]), 0.2)
print(x)
#print(y)
print(z)
print(w)

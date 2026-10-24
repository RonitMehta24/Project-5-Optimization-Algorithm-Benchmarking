import numpy as np

class Optimization:
    def __init__(self, objective, gradient, hessian):
        self.objective = objective 
        self.gradient = gradient
        self.hessian = hessian

    def steepest_descent(self):
        pass

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
print(y)
print(z)
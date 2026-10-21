import numpy as np

class Optimization:
    def __init__(self, objective, gradient, hessian):
        self.objective = objective 
        self.gradient = gradient
        self.hessian = hessian

    def steepest_descent(self):
        pass

    def newton_method(self):
        pass

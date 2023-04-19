# Numerical investigation of properties of quadrature rules
# Example: rate of convergence for composite Gauss-Legendre quadrature

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', size=16)

# We can use the quadrature() function from the Week 6 tutorial
def quadrature(f, xk, wk, a, b):
    '''
    Approximates the integral of f over [a, b],
    using the quadrature rule with weights wk
    and nodes xk.
    
    Input:
    f (function): function to integrate (as a Python function object)
    xk (Numpy array): vector containing all nodes
    wk (Numpy array): vector containing all weights
    a (float): left boundary of the interval
    b (float): right boundary of the interval
    
    Returns:
    I_approx (float): the approximate value of the integral
        of f over [a, b], using the quadrature rule.
    '''
    # Define the shifted and scaled nodes
    yk = (b - a)/2 * (xk + 1) + a
    
    # Compute the weighted sum
    I_approx = (b - a)/2 * np.sum(wk * f(yk))
    
    return I_approx


# The nodes are defined as the roots of the n-th degree Legendre polynomial.
# Calculate nodes and weights for N nodes
N = 4
xk, wk = np.polynomial.legendre.leggauss(N)

# Let's choose an arbitrary function (not a polynomial) with a known integral
def f(x):
    return np.arctan(x)

def F(x):
    '''
    Exact value for the indefinite integral of f(x) = atan(x).
    '''
    return x * np.arctan(x) - 0.5 * np.log(1 + x**2)


# Compute the integral over [0, 3]
a, b = 0, 3

# Create different values of h
M_vals = np.logspace(1, 10, 10, base=2, dtype=int)
# print(M)
h_vals = (b - a) / M_vals

# Get the exact integral
I_exact = F(b) - F(a)

# Get approximation
I_approx = []
for M in M_vals:
    # Get boundaries of all sub-intervals
    bounds = np.linspace(a, b, M+1)

    # Apply the quadrature rule over each sub-interval
    I = 0
    for i in range(M):
        I += quadrature(f, xk, wk, bounds[i], bounds[i+1])
    
    I_approx.append(I)

error = np.abs(I_exact - I_approx)

# Plot the results
fig, ax = plt.subplots()
ax.plot(h_vals, error, 'rx')
ax.set(xlabel='h', ylabel='Error', title=f'Convergence rate for GL quadrature with N={N}')
ax.set(xscale='log', yscale='log')
plt.show()

# Get the value of the convergence rate
# Select valid values of the error
threshold = 1e-14
h_vals = h_vals[error >= threshold]
error = error[error >= threshold]
r, _ = np.polyfit(np.log(h_vals), np.log(error), 1)
print(f'The rate of convergence is r = {r:.4g}')
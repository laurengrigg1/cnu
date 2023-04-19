# Finite difference approximations to derivatives
# Estimation of order of accuracy

import numpy as np
import matplotlib.pyplot as plt

# Define function for f(x) and f'(x)
def f(x):
    return np.arctan(x)

def f_prime(x):
    return 1 / (1 + x**2)


def Dc(f, x, dx):
    '''
    Centred difference
    '''
    return (f(x + dx) - f(x - dx)) / (2 * dx)


# Choose a point at which to estimate the derivative
x0 = 1.5

# Get different values of dx
dx_vals = np.logspace(-2, -16, 15, base=2)
# print(dx_vals)

# Get errors for different values of dx
error = []
for dx in dx_vals:
    f_prime_approx = Dc(f, x0, dx)
    error.append(np.abs(f_prime_approx - f_prime(x0)))

# Plot the results
fig, ax = plt.subplots()
ax.plot(dx_vals, error, 'rx')
ax.set(xlabel='h', ylabel='Error', title=f'Order of accuracy of centred difference')
ax.set(xscale='log', yscale='log')
plt.show()

# Get the value of the convergence rate
# Select valid values of the error
threshold = 1e-14
error = np.array(error)
dx_vals = dx_vals[error >= threshold]
error = error[error >= threshold]
k, _ = np.polyfit(np.log(dx_vals), np.log(error), 1)
print(f'The order of accuracy is k = {k:.4g}')
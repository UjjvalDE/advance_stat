import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Given data points (x, y)
data_points = [
    (1, 20.85), (14, 1622627797726.91), (6, 329121253.59), (19, 35371559790395.33),
    (-4, 8572074.16), (-8, 7574928255.56), (18, 20731638272229.04), (-10, 66518434081.6),
    (17, 11263436507068.29), (15, 3389201608113.09), (-14, 1900972956638.5),
    (-15, 3738722005965.96), (4, 5356952.6), (-16, 6704515113323.65),
    (-19, 39811691031504.62), (-6, 433594479.14), (-11, 179198697745.6),
    (-17, 12885964218805.07), (7, 1551009285.23), (13, 758110112057.32),
    (0, -7.89), (3, 297191.82)
]

# Separate x and y values
x_vals, y_vals = zip(*data_points)
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

# Create the design matrix for a 15th degree polynomial
X = np.vander(x_vals, 16, increasing=True)  # 16 columns for degree 0 to 15

# 1. OLS solution (no regularization)
# Solve for alpha using the normal equation
alpha_ols = np.linalg.pinv(X.T @ X) @ X.T @ y_vals
print("OLS coefficients:", alpha_ols)

# 2. Ridge Regularization
lambda_val = 1e3 
# Using sklearn's Ridge regression
ridge_model = Ridge(alpha=lambda_val, fit_intercept=False)
ridge_model.fit(X, y_vals)
alpha_ridge = ridge_model.coef_
print("Ridge Regularized coefficients:", alpha_ridge)

# 3. Model Quality Assessment
# Compute Mean Squared Error (MSE) for both models
mse_ols = np.mean((y_vals - X @ alpha_ols) ** 2)
mse_ridge = np.mean((y_vals - X @ alpha_ridge) ** 2)
print("MSE (OLS):", mse_ols)
print("MSE (Ridge):", mse_ridge)

# Prepare values for plotting
x_plot = np.linspace(min(x_vals), max(x_vals), 500)
X_plot = np.vander(x_plot, 16, increasing=True)

# Predicted values
y_ols_pred = X_plot @ alpha_ols
y_ridge_pred = X_plot @ alpha_ridge

# Plotting
plt.figure(figsize=(12, 6))

# Original data points
plt.scatter(x_vals, y_vals, color='black', label='Data points')

# OLS fit
plt.plot(x_plot, y_ols_pred, color='blue', label='OLS Fit', linestyle='--')

# Ridge fit
plt.plot(x_plot, y_ridge_pred, color='red', label='Ridge Regularized Fit', linestyle='-')

# Labels and legend
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Polynomial Fit with OLS and Ridge Regularization")
plt.legend()
plt.yscale("symlog") 
plt.grid(True)
plt.show()

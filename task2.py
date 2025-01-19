import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Given parameters
xi5 = 0.39
xi6 = 4
xi7 = 0.60
xi8 = 7

# Function to calculate survival probability P(wait > y)
def survival_function(y):
    return xi5 * np.exp(-xi6 * y) + xi7 * np.exp(-xi8 * y)

# Probability of waiting between 2 and 4 hours
prob_between_2_and_4 = survival_function(2) - survival_function(4)

# Display the probability between 2 and 4 hours
print(f"Probability of waiting between 2 and 4 hours: {prob_between_2_and_4:.4f}")

# Define the probability density function (PDF)
def pdf(y):
    # Derivative of survival function to get PDF
    return xi5 * xi6 * np.exp(-xi6 * y) + xi7 * xi8 * np.exp(-xi8 * y)

# Plot the probability density function (PDF)
y_vals = np.linspace(0, 10, 1000)
pdf_vals = pdf(y_vals)

plt.figure(figsize=(10, 6))
plt.plot(y_vals, pdf_vals, label="PDF of waiting time", color='blue')
plt.title("Probability Density Function (PDF) of Waiting Time")
plt.xlabel("Time (hours)")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()

# Generate a histogram for waiting times in minutes (1 minute intervals)
minutes_vals = np.arange(0, 240, 1) / 60  # Converting minutes to hours
hist_vals = pdf(minutes_vals)

plt.figure(figsize=(10, 6))
plt.bar(minutes_vals, hist_vals, width=0.01, color='green', alpha=0.6)
plt.title("Histogram of Waiting Time (per minute)")
plt.xlabel("Time (hours)")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# Now we calculate the mean, variance, and quartiles

# Function to compute the mean using numerical integration
mean, _ = quad(lambda y: y * pdf(y), 0, np.inf)

# Function to compute the variance using numerical integration
variance, _ = quad(lambda y: (y - mean)**2 * pdf(y), 0, np.inf)

# Quartiles
q1, _ = quad(lambda y: pdf(y), 0, mean)  # First quartile approximation
q3, _ = quad(lambda y: pdf(y), mean, np.inf)  # Third quartile approximation

# Display the computed statistics
print(f"Mean waiting time: {mean:.4f} hours")
print(f"Variance of waiting time: {variance:.4f} hours^2")
print(f"1st Quartile (Q1): {q1:.4f}")
print(f"3rd Quartile (Q3): {q3:.4f}")

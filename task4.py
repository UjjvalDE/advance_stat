import numpy as np
from scipy import stats

# Given values
xi11 = 912  # Historical mean weight (μ)
xi12 = 20.7  # Historical standard deviation (σ)
xi13 = 0  # Not used in this context
xi14 = [817, 891, 801, 897, 1020, 810, 940, 885, 903, 875]  # Sample weights from the new system

# Hypotheses
# H0: μ >= 912 (The mean weight of hammers produced by the new system is equal to or greater than the historical mean weight)
# H1: μ < 912 (The mean weight of hammers produced by the new system is lower than the historical mean)

# Sample size
n = len(xi14)

# Sample mean
sample_mean = np.mean(xi14)

# Standard deviation of the sample
sample_std_dev = np.std(xi14, ddof=1)

# Test statistic (t)
t_statistic = (sample_mean - xi11) / (sample_std_dev / np.sqrt(n))

# Degrees of freedom
df = n - 1

# Critical t-value for a one-tailed test with α=0.05
alpha = 0.05
critical_t_value = stats.t.ppf(alpha, df)

# Print results
print(f"Sample Mean: {sample_mean} grams")
print(f"t-Statistic: {t_statistic}")
print(f"Critical t-Value: {critical_t_value}")

# Conclusion
if t_statistic < critical_t_value:
    print("Reject the null hypothesis: The new system produces hammers with a lower mean weight than the historical mean.")
else:
    print("Fail to reject the null hypothesis: There is no sufficient statistical evidence to conclude that the new system produces hammers with a lower mean weight than the historical mean.")
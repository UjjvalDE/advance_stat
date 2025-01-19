import matplotlib.pyplot as plt

# Given parameters
p_for = 0.81  # Probability of vote = "for"
p_against = 1 - p_for  # Probability of vote = "against"

# Create a bar plot representing the proportions of "for" and "against"
votes = ['For', 'Against']
proportions = [p_for, p_against]

# Plotting the bar chart
plt.figure(figsize=(6, 4))
plt.bar(votes, proportions, color=['green', 'red'], alpha=0.7)
plt.title('Proportion of Votes in a Bernoulli Trial')
plt.xlabel('Vote Outcome')
plt.ylabel('Proportion')
plt.ylim(0, 1)

# Add percentage labels on top of the bars
for i in range(len(votes)):
    plt.text(i, proportions[i] + 0.02, f'{proportions[i]*100:.1f}%', ha='center', color='black')

# Display the plot
plt.show()

# Expectation of a Bernoulli distribution is just the probability of success (p_for)
expectation = p_for
print(f"The expectation of the vote outcome (P(vote = 'for')) is: {expectation:.2f}")
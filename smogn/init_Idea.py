import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#1.	Log-Normal-Like Distribution: The skewed_distribution function uses an exponential decay based on the distance from the focus index. This simulates a skewed distribution.
#2.	Steepness Parameter: The steepness_param controls how quickly the values drop off as you move away from the focus point. A higher value will create a sharper drop, while a lower value will make the distribution flatter.
#3.	Skewness Parameter: The skew_param adjusts the level of asymmetry in the distribution. Lower values make the distribution more skewed, while higher values approach uniformity.



# Define the categories and total population
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
target_population = 4000

# Parameters for skewness and steepness
focus_index = 0  # Focus on 'A' (index 0)
skew_param = 0.6  # Controls skewness (the lower, the steeper)
steepness_param = 2  # Controls steepness

# Function to generate a skewed distribution using log-normal
def skewed_distribution(focus_index, categories, skew_param, steepness_param):
    x = np.arange(len(categories)) - focus_index
    # Use a log-normal distribution centered at the focus index
    distribution = np.exp(-steepness_param * np.abs(x) ** skew_param)
    distribution /= distribution.sum()  # Normalize to sum to 1
    return distribution

# Generate the population distribution based on the skewed function
weights = skewed_distribution(focus_index, categories, skew_param, steepness_param)
population_distribution = (weights * target_population).astype(int)

# Print the results
for cat, pop in zip(categories, population_distribution):
    print(f"{cat}: {pop}")

# Plot the distribution
sns.barplot(x=categories, y=population_distribution)
plt.title(f"Skewed Distribution with focus on '{categories[focus_index]}'")
plt.show()
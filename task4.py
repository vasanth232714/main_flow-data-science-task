import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the CSV file
filename = "C:\\Users\\lmxiv\\Downloads\\USvideos.csv"  
df = pd.read_csv(filename)

# Step 2: Display basic information about the dataset
print("Basic Information:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Handle Missing Values
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Step 4: Visualize the Distribution of Variables
print("\nVisualizing Distribution of Variables:")
df.hist(bins=30, figsize=(15, 10))
plt.suptitle("Distribution of Variables", fontsize=16)
plt.show()

# Step 5: Identify Outliers
print("\nBoxplot to Identify Outliers:")
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplot of Variables", fontsize=16)
plt.show()

# Step 6: Check for Correlations Between Variables
print("\nCorrelation Heatmap:")
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# Optional: Pairplot to visualize relationships between variables
print("\nPairplot of Variables:")
sns.pairplot(df)
plt.suptitle("Pairplot of Variables", fontsize=16)
plt.show()

# Save the cleaned or processed dataset if needed
df.to_csv('processed_dataset.csv', index=False)
print("Processed dataset saved as 'processed_dataset.csv'")

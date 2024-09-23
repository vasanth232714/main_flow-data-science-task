# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy import stats

# Load dataset (replace 'your_dataset.csv' with the actual dataset file)
df = pd.read_csv('your_dataset.csv')

# ==========================================
# 1. Data Cleaning
# ==========================================

# Checking for missing values
print("Missing values before imputation:\n", df.isnull().sum())

# Handle missing values using SimpleImputer (mean for numerical columns)
imputer = SimpleImputer(strategy='mean')
df['column_name'] = imputer.fit_transform(df[['column_name']])  # Modify for your specific column(s)

# Remove outliers using Z-score method for numerical columns (e.g., 'age')
df = df[(np.abs(stats.zscore(df['age'])) < 3)]  # Adjust column as per your dataset

# Handling categorical inconsistencies, e.g., standardizing 'gender' data
df['gender'] = df['gender'].replace({'M': 'Male', 'F': 'Female'})  # Modify based on dataset

# Checking again for missing values
print("Missing values after cleaning:\n", df.isnull().sum())

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================

# Check the general statistics of the dataset
print(df.describe())

# Check the data distribution for a key column (e.g., 'age')
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualize the correlation matrix to find relationships between variables
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# ==========================================
# 3. Question Formulation and Solving
# ==========================================
# Develop and answer 7 questions based on the dataset:

# 1. What is the correlation between age and cholesterol levels?
correlation_age_chol = df['age'].corr(df['cholesterol'])  # Replace with actual columns
print(f"Correlation between Age and Cholesterol: {correlation_age_chol:.2f}")

# 2. Do men or women have a higher likelihood of heart disease (or another condition)?
gender_heart_disease = df.groupby('gender')['heart_disease'].mean()  # Modify for your dataset
print("Heart Disease Likelihood by Gender:\n", gender_heart_disease)

# 3. What is the distribution of cholesterol levels in people with and without heart disease?
sns.boxplot(x='heart_disease', y='cholesterol', data=df)  # Replace with actual columns
plt.title('Cholesterol Distribution by Heart Disease Status')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Cholesterol Level')
plt.show()

# 4. How does blood pressure vary with age?
sns.scatterplot(x='age', y='blood_pressure', data=df)  # Replace with actual columns
plt.title('Blood Pressure vs Age')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()

# 5. What are the top 3 factors most associated with heart disease?
correlation_with_heart_disease = df.corr()['heart_disease'].sort_values(ascending=False)  # Adjust as per dataset
print("Top 3 factors associated with heart disease:\n", correlation_with_heart_disease[1:4])

# 6. Does the likelihood of heart disease increase with higher cholesterol levels?
sns.lmplot(x='cholesterol', y='heart_disease', data=df, logistic=True)  # Modify as per dataset
plt.title('Cholesterol vs Likelihood of Heart Disease')
plt.xlabel('Cholesterol Level')
plt.ylabel('Heart Disease (0 = No, 1 = Yes)')
plt.show()

# 7. How do different age groups contribute to the risk of heart disease?
df['age_group'] = pd.cut(df['age'], bins=[29, 40, 50, 60, 70, 80], labels=['30-40', '40-50', '50-60', '60-70', '70-80'])
age_group_heart_disease = df.groupby('age_group')['heart_disease'].mean()  # Modify as per dataset
print("Heart Disease Likelihood by Age Group:\n", age_group_heart_disease)

# ==========================================
# 4. Data Visualization
# ==========================================

# 1. Distribution of Age
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution in the Dataset')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Boxplot for Cholesterol levels by Heart Disease status
sns.boxplot(x='heart_disease', y='cholesterol', data=df)
plt.title('Cholesterol Levels by Heart Disease Status')
plt.xlabel('Heart Disease')
plt.ylabel('Cholesterol Level')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 4. Blood Pressure vs Age Scatter Plot
sns.scatterplot(x='age', y='blood_pressure', data=df)
plt.title('Blood Pressure vs Age')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()

# 5. Bar chart for Heart Disease Likelihood by Age Group
age_group_heart_disease.plot(kind='bar', color='lightblue')
plt.title('Heart Disease Likelihood by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Likelihood of Heart Disease')
plt.show()

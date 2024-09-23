# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from scipy import stats

# Load the dataset
# Ensure you have the dataset in the same directory or specify the correct path
df = pd.read_csv('heart_disease_data.csv')

# ==========================================
# 1. Data Cleaning
# ==========================================

# Checking for missing values
print("Missing values before imputation:\n", df.isnull().sum())

# Impute missing values in 'cholesterol' using mean
imputer = SimpleImputer(strategy='mean')
df['cholesterol'] = imputer.fit_transform(df[['cholesterol']])

# Checking again for missing values after imputation
print("Missing values after imputation:\n", df.isnull().sum())

# Remove outliers using Z-score method for 'age'
df = df[(np.abs(stats.zscore(df['age'])) < 3)]

# Convert gender data to standardized format (if needed)
df['gender'] = df['gender'].replace({'M': 'Male', 'F': 'Female'})

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================

# Display summary statistics of the dataset
print(df.describe())

# Check the distribution of 'age'
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ==========================================
# 3. Question Formulation and Solving
# ==========================================

# 1. What is the correlation between age and cholesterol levels?
correlation_age_chol = df['age'].corr(df['cholesterol'])
print(f"Correlation between Age and Cholesterol: {correlation_age_chol:.2f}")

# 2. Do men or women have a higher likelihood of heart disease?
gender_heart_disease = df.groupby('gender')['heart_disease'].mean()
print("Heart Disease Likelihood by Gender:\n", gender_heart_disease)

# 3. What is the distribution of cholesterol levels in people with and without heart disease?
sns.boxplot(x='heart_disease', y='cholesterol', data=df)
plt.title('Cholesterol Distribution by Heart Disease Status')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Cholesterol Level')
plt.show()

# 4. How does blood pressure vary with age?
sns.scatterplot(x='age', y='blood_pressure', data=df)
plt.title('Blood Pressure vs Age')
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.show()

# 5. What are the top 3 factors most associated with heart disease?
correlation_with_heart_disease = df.corr()['heart_disease'].sort_values(ascending=False)
print("Top 3 factors associated with heart disease:\n", correlation_with_heart_disease[1:4])

# 6. Does the likelihood of heart disease increase with a higher BMI?
sns.lmplot(x='bmi', y='heart_disease', data=df, logistic=True)
plt.title('BMI vs Likelihood of Heart Disease')
plt.xlabel('BMI')
plt.ylabel('Heart Disease (0 = No, 1 = Yes)')
plt.show()

# 7. How do different age groups contribute to the risk of heart disease?
df['age_group'] = pd.cut(df['age'], bins=[29, 40, 50, 60, 70, 80], labels=['30-40', '40-50', '50-60', '60-70', '70-80'])
age_group_heart_disease = df.groupby('age_group')['heart_disease'].mean()
print("Heart Disease Likelihood by Age Group:\n", age_group_heart_disease)

# ==========================================
# 4. Data Visualization
# ==========================================

# 1. Age distribution plot
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution in Heart Disease Dataset')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Cholesterol Levels by Heart Disease
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

# 5. Heart Disease Likelihood by Age Group
age_group_heart_disease.plot(kind='bar', color='skyblue')
plt.title('Heart Disease Likelihood by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Likelihood of Heart Disease')
plt.show()


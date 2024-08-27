import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
# Load dataset
data = pd.read_csv("D:\\trisha\\mainflow\\DATA3.csv")

# Display the first few rows of the dataset
print(data.head())

# Data Cleaning: Handle missing values using backward fill (as an example)
data.fillna(method='bfill', inplace=True)

# Convert categorical variables to numeric using Label Encoding
label_encoders = {}
categorical_columns = ['nativeLanguage', 'gender', 'education', 'city', 'country', 'section', 'cue']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define features and target variable
# Assuming we want to predict responses based on features
X = data.drop(['responseID', 'R1', 'R2', 'R3'], axis=1)  # Features
y = data[['R1', 'R2', 'R3']]  # Targets (multi-label classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)

# Initialize the Random Forest Classifier with a different random state
model = RandomForestClassifier(random_state=21)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['R1', 'R2', 'R3']))

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [60, 120, 180],
    'max_depth': [15, 25, 35],
    'min_samples_split': [3, 6, 9],
    'min_samples_leaf': [2, 3, 5]
}

y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score for each target:")
for i, col in enumerate(y.columns):
    print(f"{col}: {accuracy_score(y_test[col], y_pred[:, i])}")

# Hamming Loss: Measures the fraction of incorrect labels
print("\nHamming Loss:")
print(hamming_loss(y_test, y_pred))

# Classification Report for each label
for i, col in enumerate(y.columns):
    print(f"\nClassification Report for {col}:")
    print(classification_report(y_test[col], y_pred[:, i]))

# If you want a single accuracy score for the entire prediction
exact_match_ratio = (y_pred == y_test.values).all(axis=1).mean()
print(f"\nExact Match Ratio (Accuracy for the whole prediction): {exact_match_ratio}")
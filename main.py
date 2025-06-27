# 🚀 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 🚀 Load Dataset
data = pd.read_csv('data/student-mat.csv', sep=';')

# 🚀 Preview Data
print("\n🔍 Preview Data:")
print(data.head())

# 🚀 Plot Distribution of Final Grades (G3)
plt.figure(figsize=(8,6))
sns.histplot(data['G3'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Final Grade')
plt.ylabel('Count')
plt.show()


# 🚀 Check Categorical Columns
categorical_cols = data.select_dtypes(include=['object']).columns
print("\n🧠 Categorical Columns:\n", categorical_cols.tolist())

# 🚀 Check Numerical Columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
print("\n🧠 Numerical Columns:\n", numerical_cols.tolist())


# 🚀 Data Preprocessing: One-hot Encode Categorical Variables
data_encoded = pd.get_dummies(data, drop_first=True)
print("\n✅ Data after Encoding:\n")
print(data_encoded.head())
print("\nShape of Data:", data_encoded.shape)


# 🚀 Define Features (X) and Target (y)
X = data_encoded.drop('G3', axis=1)
y = data_encoded['G3']
print("\n📊 X Shape:", X.shape)
print("🎯 y Shape:", y.shape)


# 🚀 Check for Missing Values and Duplicates
print("\n🔎 Missing Values per Column:\n")
print(data.isnull().sum())

print("\n🔎 Total Missing Values:", data.isnull().sum().sum())

print("\n🔍 Duplicate Rows:", data.duplicated().sum())


# 🚀 Boxplot for Outlier Detection
plt.figure(figsize=(10,6))
sns.boxplot(data=data[['G1', 'G2', 'G3', 'absences']])
plt.title('Boxplot for Grades and Absences')
plt.show()


# 🚀 Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✅ Data Split Complete")
print("Training Set Shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)


# 🚀 Initialize and Train the Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print("\n✅ Model Training Complete")


# 🚀 Make Predictions
y_pred = model.predict(X_test)
print("\n✅ Predictions Made")


# 🚀 Plot Actual vs Predicted Grades
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.grid(True)

# 🚀 Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation Complete")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

plt.show()
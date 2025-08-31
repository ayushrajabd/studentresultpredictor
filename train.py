import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Load dataset
df = pd.read_csv("student_performance.csv")
print("Dataset Head:")
print(df.head())

# Features (X) and Target (y)
X = df[["Attendance", "StudyHours", "PastScore"]]
y = df["FinalScore"]

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

print("\nModel Performance:")
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# Save the best model (Random Forest usually better)
joblib.dump(rf, "student_model.pkl")
print("\n Model saved as student_model.pkl")

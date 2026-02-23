import polars as pl
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("🤖 TRAINING RANDOM FOREST MODEL...")

# Load cleaned data
df = pl.read_csv("data/cleaned_data.csv")
print(f"📊 Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Features & target
features = ["tenure", "MonthlyCharges", "TotalCharges", "age", "years_customer", "total_spent", "avg_monthly"]
X = df.select(features).to_numpy()
y = df["Churn"].to_numpy()

print(f"🎯 Features: {len(features)}, Churn rate: {y.mean():.1%}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest (production-grade)
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n📈 MODEL PERFORMANCE REPORT:")
print(classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
importance_df = pl.DataFrame({
    "feature": features,
    "importance": importances
}).sort("importance", descending=True)

print("\n🏆 TOP FEATURES:")
print(importance_df)

# Save model
joblib.dump(model, "data/churn_model.pkl")
print("\n✅ MODEL SAVED: data/churn_model.pkl")
print("🎉 ML TRAINING COMPLETE!")

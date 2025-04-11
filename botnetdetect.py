import lightgbm as lgb
import pandas as pd
import sys

# Load model
model = lgb.Booster(model_file="Model_LightGBM.txt")

# Load test data
test_data = pd.read_csv("dataset/test.csv")
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Make predictions
predictions = model.predict(X_test)
predictions = [1 if pred >= 0.5 else 0 for pred in predictions]

# Save results
results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
results.to_csv("results.txt", index=False)

print("Detection complete. Results saved to results.txt")

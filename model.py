import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def train_lightgbm_model():
    # Load training dataset
    train_data = pd.read_csv("dataset/train.csv")
    X_train_full = train_data.iloc[:, :-1].copy()  # All features
    y_train_full = train_data.iloc[:, -1]  # Labels

    # Load test dataset
    test_data = pd.read_csv("dataset/test.csv")
    X_test = test_data.iloc[:, :-1].copy()  # Features
    y_test = test_data.iloc[:, -1]  # Labels

    # Encode string columns in both train and test sets
    le = LabelEncoder()
    for column in X_train_full.columns:
        if X_train_full[column].dtype == 'object':
            # Fit encoder on combined train and test data
            combined = pd.concat([X_train_full[column], X_test[column]], axis=0).astype(str)
            le.fit(combined)
            X_train_full.loc[:, column] = le.transform(X_train_full[column].astype(str))
            X_test.loc[:, column] = le.transform(X_test[column].astype(str))

    # Split training data into train and validation sets for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    # Model parameters with adjustments to reduce overfitting
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'learning_rate': 0.01,
        'num_leaves': 15,  # Reduced from 31 to limit tree complexity
        'min_data_in_leaf': 20,  # Prevent over-specific splits
        'is_unbalance': True,
        'verbose': -1  # Suppress detailed logs
    }

    # Train model with early stopping using callbacks
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,  # Increased for early stopping
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]  # Early stopping and no log
    )

    # Save model
    model.save_model("Model_LightGBM.txt")

    # Make predictions on test set
    y_pred_proba = model.predict(X_test)
    y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]  # Convert to binary predictions

    # Calculate accuracy and detailed metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model training complete. Saved to Model_LightGBM.txt")
    print(f"Accuracy on test set: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    model = train_lightgbm_model()
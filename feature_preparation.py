import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/extracted_features.csv")

# Load labels
labels = pd.read_csv("dataset/labels.csv")

# Debug: Check shapes and contents
print("Shape of extracted_features.csv:", df.shape)
print("Columns in extracted_features.csv:", df.columns.tolist())
print("Shape of labels.csv:", labels.shape)
print("First few rows of labels.csv:", labels.head())

# Check for row mismatch
if len(df) != len(labels):
    print(f"Warning: Number of rows in extracted_features.csv ({len(df)}) does not match labels.csv ({len(labels)})")
    # Option 1: Truncate labels to match df (not recommended, fix source instead)
    # labels = labels.iloc[:len(df)]
    # Option 2: Raise error to force fix
    raise ValueError(f"Number of rows in extracted_features.csv ({len(df)}) does not match labels.csv ({len(labels)}). Please regenerate extracted_features.csv with a matching PCAP file.")

df["label"] = labels["label"]

# Encode all string columns
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df.loc[:, column] = le.fit_transform(df[column].astype(str))

# Use all features (including tcp_flags) for consistency
all_feature_columns = [col for col in df.columns if col != 'label']
X = df[all_feature_columns]
y = df["label"]  # Use merged label column

# Verify shapes before splitting
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
if len(X) != len(y):
    raise ValueError(f"Inconsistent number of samples: X has {len(X)} rows, y has {len(y)} rows")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save to CSV with all features
train_data = X_train.copy()
train_data["label"] = y_train
train_data.to_csv("dataset/train.csv", index=False)

test_data = X_test.copy()
test_data["label"] = y_test
test_data.to_csv("dataset/test.csv", index=False)

print("Dataset prepared. Training and testing data saved with all features.")
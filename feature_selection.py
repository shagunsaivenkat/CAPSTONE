import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset/extracted_features.csv")

# Load labels and merge
labels = pd.read_csv("dataset/labels.csv")
if len(df) != len(labels):
    raise ValueError(f"Number of rows in extracted_features.csv ({len(df)}) does not match labels.csv ({len(labels)}). Please regenerate extracted_features.csv with a matching PCAP file.")
df["label"] = labels["label"]

# Debug: Check shapes and columns
print("Shape of df after merge:", df.shape)
print("Columns in df:", df.columns.tolist())

# Encode string columns before feature selection
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object' and column != 'label':
        df.loc[:, column] = le.fit_transform(df[column].astype(str))

# Ensure X and y have the same number of samples
X = df.drop(columns=["label"])
y = df["label"]

# Verify shapes before splitting
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
if len(X) != len(y):
    raise ValueError(f"Inconsistent number of samples: X has {len(X)} rows, y has {len(y)} rows")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Selection (select top 5 features, but we'll use your specified ones)
selector = SelectKBest(f_classif, k=8)
X_new = selector.fit_transform(X_train, y_train)

# Get selected feature indices
selected_feature_indices = selector.get_support()

# Use the specified features instead of dynamic selection for consistency
specified_features = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packet_count', 'total_bytes', 'tcp_flags']
selected_features = [col for col in specified_features if col in X.columns]

# Save selected features in the specified format
with open("Feature_Selection_Results.txt", "w") as f:
    f.write("Type\tm\tfeatures\n")
    f.write(f"Common\t-\t{', '.join(selected_features)}\n")

print("Feature selection completed. Selected features saved in Feature_Selection_Results.txt")
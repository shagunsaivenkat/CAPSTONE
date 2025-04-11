import pandas as pd

# Load the extracted features
df = pd.read_csv("dataset/extracted_features.csv")

# Print summary to understand the data
print("Data Summary:")
print(df.describe())
print("\nUnique Protocols:", df['protocol'].unique())
print("\nUnique TCP Flags:", df['tcp_flags'].unique())
print("\nPorts Range:")
print("Source Ports:", df['src_port'].min(), "-", df['src_port'].max())
print("Destination Ports:", df['dst_port'].min(), "-", df['dst_port'].max())

# Example labeling logic
labels = []

for index, row in df.iterrows():
    label = 0  # Default to normal (0)

    # Suspicious criteria
    if row['protocol'] in [17, 1]:  # UDP or ICMP (often used in attacks)
        label = 1
    elif row['src_port'] > 1024 and row['dst_port'] > 1024 and row['packet_count'] < 5:  # Potential port scanning
        label = 1
    elif row['total_bytes'] > 50000:  # Very large data transfer (potential DoS or exfiltration)
        label = 1
    elif row['tcp_flags'] in ['S', 'FA'] and row['packet_count'] < 3:  # Suspicious single SYN or FIN/ACK
        label = 1
    

    labels.append(label)

# Create labels DataFrame
labels_df = pd.DataFrame({"label": labels})

# Save to CSV
labels_df.to_csv("dataset/labels.csv", index=False)

print("labels.csv created in dataset/ directory with", len(labels), "labels.")
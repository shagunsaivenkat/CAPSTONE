from flask import Flask, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
import subprocess
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model import train_lightgbm_model
import time

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = "uploads/"
app.config['DATASET_FOLDER'] = "dataset/"

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pcap'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Train or load the model once when the app starts
model = train_lightgbm_model()
print("Model loaded or trained successfully.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Generate a unique filename for the new extracted features
                unique_filename = f"uploaded_features_{int(time.time())}.csv"
                uploaded_features_path = os.path.join(app.config['DATASET_FOLDER'], unique_filename)

                # Extract features from the uploaded PCAP and save to a new file
                # Use a separate log file to avoid corrupting the CSV
                log_path = os.path.join(app.config['DATASET_FOLDER'], f"extraction_log_{int(time.time())}.txt")
                with open(log_path, 'w') as log_file:
                    # Ensure Feature_Extraction.py writes to the new file
                    subprocess.run(["python", "Feature_Extraction.py", file_path, uploaded_features_path], check=True, stdout=log_file, stderr=subprocess.STDOUT)

                # Load the extracted features from the new file
                df = pd.read_csv(uploaded_features_path)
                print("Columns in uploaded_features.csv:", df.columns.tolist())  # Debug
                print("Dtypes in uploaded_features.csv:", df.dtypes)  # Debug dtypes

                # Load train data to check existing encodings
                train_data = pd.read_csv("dataset/train.csv")
                print("Columns in train.csv:", train_data.columns.tolist())  # Debug
                print("Dtypes in train.csv:", train_data.dtypes)  # Debug dtypes

                # Encode string columns
                le = LabelEncoder()
                for column in df.columns:
                    if df[column].dtype == 'object':
                        if column in train_data.columns:
                            combined = pd.concat([train_data[column].astype(str), df[column].astype(str)], axis=0)
                            try:
                                le.fit(combined)
                                df.loc[:, column] = le.transform(df[column].astype(str))
                            except ValueError as e:
                                flash(f"Encoding error for column '{column}': {e}. Using -1 for unseen labels.")
                                df.loc[:, column] = le.transform(df[column].astype(str).fillna(-1))  # Handle unseen labels
                        else:
                            flash(f"Column '{column}' not found in train data, skipping encoding")
                            continue

                # Verify all columns are numerical
                if df.dtypes.any() == 'object':
                    flash(f"Non-numerical columns found: {df.select_dtypes(include=['object']).columns.tolist()}")
                    return redirect(request.url)

                # Load selected features
                with open("Feature_Selection_Results.txt", "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    print("Contents of Feature_Selection_Results.txt:", lines)  # Debug
                    common_features_str = next((line.split("\t")[2] for line in lines[1:] if line.startswith("Common")), None)
                    if not common_features_str:
                        flash("Could not find 'Common' features in Feature_Selection_Results.txt")
                        return redirect(request.url)
                    selected_features = [feat.strip() for feat in common_features_str.split(",")]

                # Verify selected features exist
                missing_features = [f for f in selected_features if f not in df.columns]
                if missing_features:
                    flash(f"Missing features in extracted data: {missing_features}")
                    return redirect(request.url)

                # Prepare input for prediction
                X_new = df[selected_features]

                # Make predictions
                y_pred_proba = model.predict(X_new)
                y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]

                # Determine botnet detection
                botnet_detected = any(y_pred)
                result = "Botnet Detected" if botnet_detected else "No Botnet Detected"

                return render_template('index.html', result=result, features=common_features_str)
            except subprocess.CalledProcessError as e:
                flash(f"Error processing file: {e}")
                return redirect(request.url)
            except Exception as e:
                flash(f"An unexpected error occurred: {e}")
                if 'src_ip' in str(e) or 'dst_ip' in str(e):
                    flash("Error due to non-numerical 'src_ip' or 'dst_ip'. Ensure encoding is applied.")
                return redirect(request.url)
        else:
            flash('Invalid file format. Please upload a .pcap file.')
            return redirect(request.url)

    return render_template('index.html', result=None, features=None)

if __name__ == '__main__':
    app.run(debug=True)
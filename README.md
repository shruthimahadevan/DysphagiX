
Esophageal Motility Disorder Classification
📌 Project Overview
This project focuses on the classification of esophageal motility disorders using Machine Learning (ML). The model is trained using Random Forest Classifier to predict different disorders based on clinical measurements:

Integrated Relaxation Pressure (IRP)

Distal Latency (DL)

Distal Contractile Integral (DCI)

Additionally, the model includes Solid Perfused and Water Perfused diagnoses based on IRP thresholds.

🏥 Disorders Classified
Achalasia/EGJ OO

Diffuse Esophageal Spasm

Ineffective Esophageal Motility

Absent Contractility

Hypercontractile Esophagus

Normal Esophageal Pressure

🚀 Features
✅ Preprocessing & Feature Engineering

Extracting key clinical measurements (IRP, DL, DCI)

Classification based on clinical threshold values

Solid Perfused & Water Perfused classification

✅ Model Training & Optimization

Random Forest Classifier implementation

Hyperparameter tuning with GridSearchCV

Performance evaluation using classification report & cross-validation

✅ Explainable AI (XAI) with SHAP

Feature importance analysis

Summary plots for model explainability

✅ Deployment & Integration

Saving trained model as .pkl for reuse

Flask API setup for future deployment

🛠 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/esophageal-motility-classification.git
cd esophageal-motility-classification
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Model
python train_model.py
📊 Model Training & Evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

# Load data
data = pd.read_excel('data/Chicago_data.xlsx')

# Define Features & Labels
X = data[['IRP', 'DL', 'DCI']]
y = data['Disease Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model
predictions = rf_model.predict(X_test)
print(classification_report(y_test, predictions))
print("Accuracy:", accuracy_score(y_test, predictions))

# Save Model
joblib.dump(rf_model, 'esophageal_model.pkl')
📌 SHAP Explainability
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)
shap.summary

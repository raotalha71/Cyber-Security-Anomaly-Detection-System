# Cyber Security Anomaly Detection System

A real-time anomaly detection system designed to identify cybersecurity threats in network traffic using a layered machine learning pipeline.

## Overview
This project implements a multi-stage anomaly detection system to monitor network data for potential threats. It combines unsupervised learning techniques to detect anomalies and assess their severity, making it suitable for real-time cybersecurity applications.

### Features
- **Input Options**: Upload CSV files or enter data manually via a Streamlit dashboard.
- **Pipeline**:
  1. **Isolation Forest**: Initial anomaly detection on full dataset.
  2. **One-Class SVM**: Refines anomalies from Isolation Forest.
  3. **Autoencoder**: Final severity scoring (Low/Medium/High) using a Deep Neural Network.
- **Performance**: Achieves ~81% F1 score on a 40,000-row dataset with 25 features.
- **Visualization**: Displays results and severity distribution in real-time.

## Requirements
- Python 3.11+
- Libraries: `streamlit`, `pandas`, `numpy`, `scikit-learn==1.6.1`, `tensorflow`, `joblib`

Install dependencies:
```bash
pip install -r requirements.txt

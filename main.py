import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN notice

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load trained models and scaler with full paths
iso_forest = joblib.load("D:/Projects/Cyber_attack_20000/Sources/Cyber-Security-Anomaly-Detection-System/iso_forest.pkl")
oc_svm = joblib.load("D:/Projects/Cyber_attack_20000/Sources/Cyber-Security-Anomaly-Detection-System/oc_svm.pkl")
autoencoder = tf.keras.models.load_model("D:\Projects\Cyber_attack_20000\Sources\Cyber-Security-Anomaly-Detection-System/autoencoder.keras")
scaler = joblib.load("D:/Projects/Cyber_attack_20000/Sources/Cyber-Security-Anomaly-Detection-System/scaler.pkl")

# Define the 25 feature columns
DESIRED_COLUMNS = [
    "Source Port", "Destination Port", "Packet Length", "Packet Type", "Malware Indicators",
    "Alerts/Warnings", "Firewall Logs", "IDS/IPS Alerts", "Log Source",
    "Protocol_ICMP", "Protocol_TCP", "Protocol_UDP",
    "Action_Blocked", "Action_Ignored", "Action_Logged",
    "Traffic_DNS", "Traffic_FTP", "Traffic_HTTP",
    "Attack_DDoS", "Attack_Intrusion", "Attack_Malware",
    "OS_Android", "OS_Linux", "OS_Other", "OS_Windows"
]

# Streamlit app title
st.title("Real-Time Anomaly Detection Dashboard")

# Input method selection
input_method = st.selectbox("Choose Input Method", ["Upload CSV", "Manual Input"], index=None, placeholder="Select an option...")

# Initialize user_data
user_data = None

# Process based on input method
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        # Read CSV
        data = pd.read_csv(uploaded_file)
        
        # Check for required columns
        missing_cols = [col for col in DESIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            st.error(f"CSV is missing required columns: {missing_cols}")
            st.stop()
        
        # Extract and preprocess
        data_filtered = data[DESIRED_COLUMNS]
        data_filtered = data_filtered.fillna(0)  # Handle missing values
        data_filtered["Source Port"] = data_filtered["Source Port"].clip(0, 65535)
        data_filtered["Destination Port"] = data_filtered["Destination Port"].clip(0, 65535)
        
        # Normalize the data
        user_data_scaled = scaler.transform(data_filtered.values)
        user_data = user_data_scaled[-1]  # Take the last row
        st.write("Latest Data Point from CSV (normalized):", user_data)

elif input_method == "Manual Input":
    st.subheader("Enter Data Manually")
    
    # Port numbers (0-65535)
    source_port = st.number_input("Source Port", min_value=0.0, max_value=65535.0, step=1.0)
    dest_port = st.number_input("Destination Port", min_value=0.0, max_value=65535.0, step=1.0)
    
    # Other features (0-1)
    packet_length = st.number_input("Packet Length", min_value=0.0, max_value=1.0, step=0.01)
    packet_type = st.number_input("Packet Type", min_value=0.0, max_value=1.0, step=0.01)
    malware_indicators = st.number_input("Malware Indicators", min_value=0.0, max_value=1.0, step=0.01)
    alerts_warnings = st.number_input("Alerts/Warnings", min_value=0.0, max_value=1.0, step=0.01)
    firewall_logs = st.number_input("Firewall Logs", min_value=0.0, max_value=1.0, step=0.01)
    ids_ips_alerts = st.number_input("IDS/IPS Alerts", min_value=0.0, max_value=1.0, step=0.01)
    log_source = st.number_input("Log Source", min_value=0.0, max_value=1.0, step=0.01)
    
    protocol_icmp = st.number_input("Protocol_ICMP", min_value=0.0, max_value=1.0, step=1.0)
    protocol_tcp = st.number_input("Protocol_TCP", min_value=0.0, max_value=1.0, step=1.0)
    protocol_udp = st.number_input("Protocol_UDP", min_value=0.0, max_value=1.0, step=1.0)
    
    action_blocked = st.number_input("Action_Blocked", min_value=0.0, max_value=1.0, step=1.0)
    action_ignored = st.number_input("Action_Ignored", min_value=0.0, max_value=1.0, step=1.0)
    action_logged = st.number_input("Action_Logged", min_value=0.0, max_value=1.0, step=1.0)
    
    traffic_dns = st.number_input("Traffic_DNS", min_value=0.0, max_value=1.0, step=1.0)
    traffic_ftp = st.number_input("Traffic_FTP", min_value=0.0, max_value=1.0, step=1.0)
    traffic_http = st.number_input("Traffic_HTTP", min_value=0.0, max_value=1.0, step=1.0)
    
    attack_ddos = st.number_input("Attack_DDoS", min_value=0.0, max_value=1.0, step=1.0)
    attack_intrusion = st.number_input("Attack_Intrusion", min_value=0.0, max_value=1.0, step=1.0)
    attack_malware = st.number_input("Attack_Malware", min_value=0.0, max_value=1.0, step=1.0)
    
    os_android = st.number_input("OS_Android", min_value=0.0, max_value=1.0, step=1.0)
    os_linux = st.number_input("OS_Linux", min_value=0.0, max_value=1.0, step=1.0)
    os_other = st.number_input("OS_Other", min_value=0.0, max_value=1.0, step=1.0)
    os_windows = st.number_input("OS_Windows", min_value=0.0, max_value=1.0, step=1.0)

    if st.button("Submit Manual Data"):
        user_data = np.array([
            source_port, dest_port, packet_length, packet_type, malware_indicators,
            alerts_warnings, firewall_logs, ids_ips_alerts, log_source,
            protocol_icmp, protocol_tcp, protocol_udp,
            action_blocked, action_ignored, action_logged,
            traffic_dns, traffic_ftp, traffic_http,
            attack_ddos, attack_intrusion, attack_malware,
            os_android, os_linux, os_other, os_windows
        ])
        user_data_scaled = scaler.transform([user_data])
        user_data = user_data_scaled[0]
        st.write("Manual Input (normalized):", user_data)

# Anomaly Detection Pipeline
if user_data is not None:
    # Step 1: Isolation Forest
    iso_pred = iso_forest.predict([user_data])[0]
    if iso_pred == 1:
        st.success("No Anomaly Detected")
        st.stop()
    st.write("Potential Anomaly Detected by Isolation Forest, Checking Further...")

    # Step 2: One-Class SVM
    svm_pred = oc_svm.predict([user_data])[0]
    if svm_pred == 1:
        st.success("No Threat Detected")
        st.stop()
    st.write("Anomaly Confirmed by One-Class SVM, Calculating Severity...")

    # Step 3: Autoencoder
    reconstructed = autoencoder.predict(np.array(user_data).reshape(1, -1))
    error_score = np.mean(np.power(user_data - reconstructed, 2))

    def categorize_severity(error):
        if error < 0.1:
            return "Low"
        elif error < 0.5:
            return "Medium"
        else:
            return "High"

    severity = categorize_severity(error_score)

    # Display Results
    st.subheader("Detection Results")
    st.write(f"Anomaly Status: Yes")
    st.write(f"Threat Level: {severity}")
    st.write(f"Reconstruction Error: {error_score:.4f}")

    # Simulated severity distribution (for visualization)
    severity_counts = {"Low": 5, "Medium": 3, "High": 2}
    st.bar_chart(severity_counts)

else:
    if input_method:
        st.warning("Please upload a CSV or submit manual data to proceed!")
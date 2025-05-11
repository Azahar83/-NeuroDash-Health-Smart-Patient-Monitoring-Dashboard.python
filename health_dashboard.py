import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


st.title(" NeuroDash Health: Smart Patient Monitoring Dashboard")


uploaded_file = st.file_uploader(" Upload Patient Vitals CSV", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()  

    st.success(" Data loaded successfully!")
    st.write(" Columns detected:", data.columns.tolist())


    required_columns = ["patient_id", "timestamp", "heart_rate", "oxygen_level", "temperature", "label"]
    if not all(col in data.columns for col in required_columns):
        st.error(f" Missing required columns. Expected: {required_columns}")
        st.stop()


    st.subheader(" Raw Patient Data Preview")
    st.dataframe(data.head())


    st.subheader(" Raw Data for Each Patient")
    for pid in data['patient_id'].unique():
        with st.expander(f" View Data for Patient {pid}"):
            st.dataframe(data[data['patient_id'] == pid].reset_index(drop=True))


    st.subheader(" Select Patient for Analysis")
    patient_id = st.selectbox("Choose a patient ID:", data['patient_id'].unique())
    patient_data = data[data['patient_id'] == patient_id]


    st.subheader(" Vitals Over Time")
    fig = px.line(patient_data, x="timestamp",
                  y=["heart_rate", "oxygen_level", "temperature"],
                  labels={"value": "Vital Sign", "timestamp": "Time"},
                  title=f"Vitals for Patient {patient_id}")
    st.plotly_chart(fig)


    st.subheader(" Predictive Health Risk Analysis")
    

    X = data[["heart_rate", "oxygen_level", "temperature"]]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)


    accuracy = clf.score(X_test, y_test)
    st.write(f" Model Accuracy: **{accuracy * 100:.2f}%**")


    st.subheader(" Real-time Risk Prediction")
    latest_data = patient_data[["heart_rate", "oxygen_level", "temperature"]].iloc[-1:]
    prediction = clf.predict(latest_data)[0]
    risk_status = " At Risk" if prediction == 1 else " Normal"


    st.markdown(f"""
<div style="padding:10px; background-color:{'tomato' if prediction == 1 else 'lightgreen'}; color:white; border-radius:10px">
<b>Health Status for Patient {patient_id}:</b> {risk_status}
</div>
""", unsafe_allow_html=True)



    if st.checkbox(" Show detailed model performance"):
        y_pred = clf.predict(X_test)
        st.code(classification_report(y_test, y_pred), language="text")


else:
    st.info(" Please upload a CSV file to begin.")

#for run.go terminal and run- streamlit run health_dashboard.py

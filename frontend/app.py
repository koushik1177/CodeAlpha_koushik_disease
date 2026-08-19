"""
Multi-Disease AI Diagnostic & Clinical Decision Support System.

Provides a modern medical UI/UX dashboard in Streamlit with interactive Plotly
analytics, risk diagnostic scoring, SQLite prediction history logging, and PDF reports.
"""

import sys
from pathlib import Path

# Add project root, backend, and frontend directories to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(FRONTEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.settings import DISEASE_CONFIGS, MODELS_DIR, OUTPUTS_DIR
from database.db_manager import DatabaseManager
from src.predict import DiseasePredictor
from src.data_loader import load_disease_data
from src.train import run_full_training_pipeline
from services.report_service import generate_pdf_report, generate_csv_report, generate_txt_report
from pages_modules.history_page import render_history_page


# Page Configuration
st.set_page_config(
    page_title="Disease Prediction System | AI Clinical Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Medical CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #059669 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        font-weight: 400;
        opacity: 0.95;
        margin-bottom: 0;
        color: #F1F5F9 !important;
    }

    /* High-Contrast Medical Cards */
    .med-card {
        background-color: #FFFFFF !important;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #CBD5E1;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.2rem;
        color: #0F172A !important;
    }
    .med-card h4 {
        color: #1E3A8A !important;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.4rem;
    }
    .med-card p, .med-card li, .med-card ol, .med-card span {
        color: #1E293B !important;
        font-size: 1.02rem;
        line-height: 1.6;
        font-weight: 500;
    }

    /* Custom Dropdown Selectbox Styling */
    div[data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }

    /* Risk Banners */
    .risk-banner-low {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1.5px solid #10B981;
        color: #065F46 !important;
        padding: 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .risk-banner-med {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border: 1.5px solid #F59E0B;
        color: #92400E !important;
        padding: 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1.5px solid #EF4444;
        color: #991B1B !important;
        padding: 1.5rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 1.8rem;
        margin-top: 3rem;
        border-top: 1px solid #E2E8F0;
        color: #475569 !important;
        font-size: 0.95rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    return DatabaseManager()


def check_and_train_models():
    """Auto-trains models if missing from models directory."""
    for disease_key in DISEASE_CONFIGS.keys():
        model_file = MODELS_DIR / f"model_{disease_key}.joblib"
        if not model_file.exists():
            st.info(f"Training ML models for '{disease_key}'...")
            with st.spinner("Executing GridSearch CV model training..."):
                run_full_training_pipeline()
            st.success("Model artifacts initialized!")
            break


def main():
    check_and_train_models()

    # Hero Header
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">🏥 Disease Prediction System</div>
            <div class="hero-subtitle">AI-Powered Clinical Diagnosis & Risk Analytics using Machine Learning</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation Selectbox
    st.sidebar.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
    st.sidebar.title("Medical Hub")
    
    nav_option = st.sidebar.selectbox(
        "📍 Select Module Page:",
        [
            "🩺 Predict Disease",
            "📊 Dataset Explorer",
            "📈 Visualizations",
            "🗄️ Database Inspector",
            "ℹ️ About System"
        ],
        index=0,
        key="sidebar_navigation_selectbox"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Clinical Support Notice**: AI predictions serve as decision-support assistance. Always consult a certified physician for diagnosis.")

    # Page Routing
    if nav_option == "🩺 Predict Disease":
        render_predict_page()
    elif nav_option == "📊 Dataset Explorer":
        render_dataset_page()
    elif nav_option == "📈 Visualizations":
        render_visualizations_page()
    elif nav_option == "🗄️ Database Inspector":
        render_history_page()
    elif nav_option == "ℹ️ About System":
        render_about_page()

    # Footer
    st.markdown("""
        <div class="custom-footer">
            🩺 <strong>Multi-Disease Diagnostic System</strong> | Designed with Streamlit & Scikit-Learn | Clinical AI Decision Support
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# 🩺 PREDICT DISEASE PAGE
# ==========================================
def render_predict_page():
    st.markdown("### 🩺 Clinical Risk Assessment Input")

    selected_disease = st.selectbox(
        "Select Target Disease Domain for Diagnosis:",
        ["❤️ Heart Disease", "🩸 Diabetes Risk", "🧪 Chronic Kidney Disease"]
    )

    st.markdown("---")

    if "Heart" in selected_disease:
        render_heart_inputs()
    elif "Diabetes" in selected_disease:
        render_diabetes_inputs()
    elif "Kidney" in selected_disease:
        render_kidney_inputs()


def render_heart_inputs():
    with st.form("heart_form"):
        with st.expander("👤 Patient Demographic & General Information", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_name = st.text_input("Patient Name", value="John Doe")
            with c2:
                age = st.number_input("Age (years) *", 20, 100, 52, help="Patient age in years")
            with c3:
                sex_str = st.selectbox("Sex *", options=["Male", "Female"], index=0)
                sex = 1 if sex_str == "Male" else 0

        with st.expander("🩺 Clinical Vital Measurements", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                trestbps = st.number_input("Resting Blood Pressure (mm Hg) *", 80, 220, 125, help="Normal baseline: 90-120 mm Hg")
                chol = st.number_input("Serum Cholestoral (mg/dl) *", 100, 600, 212, help="Desirable: < 200 mg/dl")
            with c2:
                thalach = st.number_input("Maximum Heart Rate Achieved *", 60, 220, 168)
                exang_str = st.selectbox("Exercise Induced Angina *", options=["No", "Yes"], index=0)
                exang = 1 if exang_str == "Yes" else 0

        with st.expander("🧪 Cardiac Diagnostic Laboratory Results", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                cp_str = st.selectbox("Chest Pain Type *", options=["0: Typical Angina", "1: Atypical Angina", "2: Non-Anginal Pain", "3: Asymptomatic"], index=0)
                cp = int(cp_str.split(":")[0])
                fbs_str = st.selectbox("Fasting Blood Sugar > 120 mg/dl *", options=["False (<= 120 mg/dl)", "True (> 120 mg/dl)"], index=0)
                fbs = 1 if "True" in fbs_str else 0
            with c2:
                restecg_str = st.selectbox("Resting ECG Results *", options=["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"], index=0)
                restecg = int(restecg_str.split(":")[0])
                oldpeak = st.number_input("ST Depression (oldpeak) *", 0.0, 6.5, 1.0, step=0.1)
            with c3:
                slope_str = st.selectbox("Slope of Peak Exercise ST *", options=["0: Upsloping", "1: Flat", "2: Downsloping"], index=0)
                slope = int(slope_str.split(":")[0])
                ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-4) *", options=[0, 1, 2, 3, 4], index=0)
                thal_str = st.selectbox("Thalassemia *", options=["1: Normal", "2: Fixed Defect", "3: Reversable Defect"], index=0)
                thal = int(thal_str.split(":")[0])

        submit_btn = st.form_submit_button("🔍 Execute Cardiac Risk Prediction", type="primary", use_container_width=True)

    if submit_btn:
        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        execute_prediction("heart", input_data, patient_name, age, sex_str)


def render_diabetes_inputs():
    st.markdown("##### 👤 Patient Demographics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        patient_name = st.text_input("Patient Name", value="Jane Doe", key="diabetes_patient_name")
    with c2:
        age = st.number_input("Age (years) *", 1, 120, 33, key="diabetes_age")
    with c3:
        sex_str = st.selectbox("Sex *", options=["Female", "Male"], index=0, key="diabetes_sex_select")
    with c4:
        if sex_str == "Female":
            pregnancies = st.number_input("Number of Pregnancies *", 0, 20, 2, key="diabetes_pregnancies")
        else:
            st.info("Pregnancies: N/A (Male)")
            pregnancies = 0

    with st.form("diabetes_form"):
        with st.expander("🩺 Metabolic & Physical Measurements", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                glucose = st.number_input("Glucose Level (mg/dL) *", 0, 300, 120, help="Normal fasting: 70-99 mg/dL")
                blood_pressure = st.number_input("Blood Pressure (mm Hg) *", 0, 150, 70)
            with c2:
                bmi = st.number_input("Body Mass Index (BMI kg/m²) *", 0.0, 70.0, 25.5, step=0.1, help="Normal range: 18.5 - 24.9")
                skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm) *", 0, 100, 20)

        with st.expander("🧪 Endocrine Laboratory Data", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                insulin = st.number_input("2-Hour Serum Insulin (mu U/ml) *", 0, 900, 79)
            with c2:
                dpf = st.number_input("Diabetes Pedigree Function Score *", 0.0, 3.0, 0.47, step=0.01)

        submit_btn = st.form_submit_button("🔍 Execute Diabetes Risk Assessment", type="primary", use_container_width=True)

    if submit_btn:
        input_data = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age
        }
        execute_prediction("diabetes", input_data, patient_name, age, sex_str)


def render_kidney_inputs():
    with st.form("kidney_form"):
        with st.expander("👤 Patient Demographics & Conditions", expanded=True):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                patient_name = st.text_input("Patient Name", value="Alex Smith")
            with c2:
                age = st.number_input("Age (years) *", 1, 120, 48)
            with c3:
                sex_str = st.selectbox("Sex *", options=["Male", "Female"], index=0)
            with c4:
                htn_str = st.selectbox("Hypertension *", options=["No", "Yes"], index=0)
                htn = 1 if htn_str == "Yes" else 0
            with c5:
                dm_str = st.selectbox("Diabetes Mellitus *", options=["No", "Yes"], index=0)
                dm = 1 if dm_str == "Yes" else 0

        with st.expander("🩺 Clinical & Renal Indicators", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                bp = st.number_input("Blood Pressure (mm Hg) *", 50, 180, 80)
                sg = st.selectbox("Specific Gravity *", options=[1.005, 1.010, 1.015, 1.020, 1.025], index=3)
                al = st.selectbox("Albumin (0-5) *", options=[0, 1, 2, 3, 4, 5], index=1)
            with c2:
                su = st.selectbox("Sugar Level (0-5) *", options=[0, 1, 2, 3, 4, 5], index=0)
                bgr = st.number_input("Blood Glucose Random (mg/dL) *", 50, 500, 121)
                bu = st.number_input("Blood Urea (mg/dL) *", 10, 400, 36)

        with st.expander("🧪 Blood & Electrolyte Laboratory Panel", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                sc = st.number_input("Serum Creatinine (mg/dL) *", 0.1, 20.0, 1.2, step=0.1, help="Normal: 0.6 - 1.2 mg/dL")
                sod = st.number_input("Sodium (mEq/L) *", 100, 180, 138)
            with c2:
                pot = st.number_input("Potassium (mEq/L) *", 2.0, 10.0, 4.4, step=0.1)
                hemo = st.number_input("Hemoglobin (g/dL) *", 3.0, 20.0, 12.5, step=0.1)
            with c3:
                pcv = st.number_input("Packed Cell Volume (%) *", 10, 60, 44)
                wbcc = st.number_input("WBC Count (cells/cumm) *", 2000, 30000, 7800)
                rbcc = st.number_input("RBC Count (millions/cumm) *", 1.0, 10.0, 5.2, step=0.1)

        submit_btn = st.form_submit_button("🔍 Execute Renal Risk Assessment", type="primary", use_container_width=True)

    if submit_btn:
        input_data = {
            "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
            "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
            "hemo": hemo, "pcv": pcv, "wbcc": wbcc, "rbcc": rbcc,
            "htn": htn, "dm": dm
        }
        execute_prediction("kidney", input_data, patient_name, age, sex_str)


def execute_prediction(disease_type: str, input_data: dict, patient_name: str, age: int, sex: str):
    with st.spinner("Executing Machine Learning Inference..."):
        try:
            predictor = DiseasePredictor(disease_type)
            result = predictor.predict_single(input_data)

            # Persist record to SQLite DB
            db = get_db()
            db.save_record(
                disease_type=disease_type,
                disease_name=result["disease"],
                patient_name=patient_name,
                age=age,
                sex=sex,
                prediction=result["prediction"],
                probability=result["probability"],
                risk_percentage=result["risk_percentage"],
                risk_category=result["risk_category"],
                status=result["status"],
                input_data=input_data
            )

            display_prediction_result_card(result, patient_name)
        except Exception as e:
            st.error(f"Inference Error: {str(e)}")


def display_prediction_result_card(result: dict, patient_name: str = "Anonymous Patient"):
    st.markdown("---")
    st.markdown("### 📋 Risk Prediction Result")

    is_high_risk = result["prediction"] == 1
    risk_pct = result["risk_percentage"]

    if risk_pct >= 70.0:
        banner_class = "risk-banner-high"
        risk_label = "HIGH RISK DETECTED"
        risk_color = "#DC2626"
        icon = "🔴"
    elif risk_pct >= 40.0:
        banner_class = "risk-banner-med"
        risk_label = "MODERATE RISK DETECTED"
        risk_color = "#D97706"
        icon = "🟡"
    else:
        banner_class = "risk-banner-low"
        risk_label = "LOW RISK / NORMAL"
        risk_color = "#059669"
        icon = "🟢"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
            <div class="{banner_class}">
                <h3>{icon} {risk_label}</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">
                    Patient <strong>{patient_name}</strong> exhibits a <strong>{risk_pct}% probability score</strong> for <strong>{result['disease']}</strong>.
                </p>
            </div>
        """, unsafe_allow_html=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diagnostic Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 40], 'color': "#D1FAE5"},
                    {'range': [40, 70], 'color': "#FEF3C7"},
                    {'range': [70, 100], 'color': "#FEE2E2"}
                ]
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("##### 🩺 Actionable Clinical Recommendations")
        if is_high_risk:
            if "Heart" in result["disease"]:
                st.warning("• Refer to Cardiology specialist immediately\n• Perform 12-Lead ECG & Echocardiogram\n• Monitor Blood Pressure & Lipid panel")
            elif "Diabetes" in result["disease"]:
                st.warning("• Schedule Fasting Plasma Glucose & HbA1c lab tests\n• Initiate dietary & insulin evaluation\n• Monitor blood glucose levels")
            else:
                st.warning("• Refer to Nephrology specialist for renal function evaluation\n• Schedule Serum Creatinine & BUN tests\n• Monitor daily Urine Albumin excretion")
        else:
            st.success("• Maintain balanced diet & physical exercise\n• Schedule routine annual checkups\n• Re-evaluate diagnostic metrics periodically")

    with col2:
        st.markdown("""<div class="med-card">""", unsafe_allow_html=True)
        st.metric(label="Diagnostic Status", value=result["status"])
        st.metric(label="Calculated Risk Score", value=f"{risk_pct}%")
        st.metric(label="Severity Classification", value=result["risk_category"])

        st.markdown("##### 📥 Export Diagnostic Reports")
        pdf_bytes = generate_pdf_report(result, patient_name)
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"Clinical_Report_{patient_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        csv_str = generate_csv_report(result, patient_name)
        st.download_button(
            "📊 Download CSV Data",
            data=csv_str,
            file_name=f"Patient_Record_{patient_name.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        txt_str = generate_txt_report(result, patient_name)
        st.download_button(
            "📝 Download TXT Summary",
            data=txt_str,
            file_name=f"Patient_Summary_{patient_name.replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 📊 DATASET EXPLORER PAGE
# ==========================================
def render_dataset_page():
    st.markdown("### 📊 Medical Dataset Explorer")

    disease_key = st.selectbox(
        "Select Dataset Domain:",
        ["heart", "diabetes", "kidney"],
        format_func=lambda x: DISEASE_CONFIGS[x]["name"]
    )

    try:
        X, y = load_disease_data(disease_key)
        df_full = X.copy()
        df_full["Target_Outcome"] = y

        st.markdown(f"**Dataset Overview: {DISEASE_CONFIGS[disease_key]['name']}**")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", df_full.shape[0])
        with c2:
            st.metric("Feature Columns", df_full.shape[1] - 1)
        with c3:
            st.metric("Positive Diagnoses", int(y.sum()))
        with c4:
            st.metric("Negative Diagnoses", int((y == 0).sum()))

        st.markdown("#### 📄 Dataset Table View")
        st.dataframe(df_full.head(10), use_container_width=True)

        st.markdown("#### 📈 Target Outcome Distribution")
        fig_pie = px.pie(
            values=y.value_counts().values,
            names=["Negative (0)", "Positive (1)"],
            title=f"Diagnosis Class Balance - {DISEASE_CONFIGS[disease_key]['name']}",
            color_discrete_sequence=["#10B981", "#EF4444"],
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")


# ==========================================
# 📈 VISUALIZATIONS PAGE
# ==========================================
def render_visualizations_page():
    st.markdown("### 📈 Interactive Visual Analytics")

    disease_key = st.selectbox(
        "Select Visualization Domain:",
        ["heart", "diabetes", "kidney"],
        format_func=lambda x: DISEASE_CONFIGS[x]["name"]
    )

    try:
        X, y = load_disease_data(disease_key)
        df_full = X.copy()
        df_full["Target"] = y

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔥 Feature Correlation Heatmap")
            corr = df_full.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="Blues",
                title=f"Correlation Matrix ({DISEASE_CONFIGS[disease_key]['name']})"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.markdown("#### 📊 Interactive Feature Distribution")
            selected_feature = st.selectbox("Select Feature to Inspect Distribution:", X.columns)
            fig_hist = px.histogram(
                df_full,
                x=selected_feature,
                color="Target",
                barmode="overlay",
                color_discrete_sequence=["#10B981", "#EF4444"],
                title=f"Distribution of '{selected_feature}' by Class"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"Error rendering visualizations: {str(e)}")


# ==========================================
# ℹ️ ABOUT PAGE
# ==========================================
def render_about_page():
    st.markdown("### ℹ️ System Architecture & Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="med-card">
                <h4>🎯 Clinical Project Purpose</h4>
                <p>The <strong>Multi-Disease Risk Diagnostic System</strong> is designed to aid clinical decision-making by leveraging supervised machine learning to identify early disease indicators for Cardiac, Diabetes, and Chronic Kidney Disease conditions.</p>
            </div>

            <div class="med-card">
                <h4>🧠 Machine Learning Architecture & Workflow</h4>
                <ol>
                    <li><strong>Data Ingestion</strong>: Clean medical feature datasets for Heart, Diabetes, and Kidney disease domains.</li>
                    <li><strong>Data Preprocessing</strong>: Missing value median imputation & StandardScaler feature scaling.</li>
                    <li><strong>Hyperparameter Tuning</strong>: 5-Fold GridSearchCV cross-validation across algorithms.</li>
                    <li><strong>Model Persistence</strong>: Joblib serialization for instant real-time inference.</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="med-card">
                <h4>🛠️ Technology Stack</h4>
                <ul>
                    <li><strong>Frontend User Interface</strong>: Streamlit, Plotly Interactive Visuals, HTML5 & CSS3</li>
                    <li><strong>Machine Learning Libraries</strong>: Scikit-Learn, Joblib, NumPy, Pandas</li>
                    <li><strong>Supervised Classifiers</strong>: Logistic Regression, Random Forest, Support Vector Machines (SVM), Gradient Boosting</li>
                    <li><strong>Database & Reporting</strong>: SQLite3 Database, FPDF PDF Reports, CSV & TXT Exporters</li>
                </ul>
            </div>

            <div class="med-card">
                <h4>✨ Key Diagnostic Features</h4>
                <ul>
                    <li>Multi-Domain Risk Evaluation (Heart, Diabetes, Kidney)</li>
                    <li>Interactive Risk Gauges & Diagnostic Probability Scores</li>
                    <li>Actionable Follow-up Clinical Guidelines</li>
                    <li>Multi-Format Reports (PDF, CSV, TXT) & SQLite History Database</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
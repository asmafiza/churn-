!pip install streamlit -q
!pip install pyngrok -q
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Page Configuration
st.set_page_config(page_title="Telco Customer Churn Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Telco Customer Churn Analytics Dashboard")
st.markdown("Is dashboard ke zariye aap customer churn data ko analyze kar sakte hain aur Machine Learning model se predictions dekh sakte hain.")

# 2. Sidebar for File Upload
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Apni Telco Churn CSV File Upload Karein", type=["csv"])

# Agar user file upload nahi karta toh sample/dummy data handle karne ke liye placeholder
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Basic Cleaning (jaisa aapke main notebook code me tha)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    # Target variable check
    target_col = 'Churn' if 'Churn' in df.columns else df.columns[-1]

    # --- ROW 1: Key Metrics ---
    st.subheader("📈 Business Key Performance Indicators (KPIs)")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    total_customers = len(df)
    churn_rate = (df[target_col].value_counts(normalize=True).get('Yes', 0) if df[target_col].dtype == 'O' else df[target_col].value_counts(normalize=True).get(1, 0)) * 100
    avg_tenure = df['tenure'].mean() if 'tenure' in df.columns else 0
    
    kpi1.metric(label="Total Customers", value=f"{total_customers:,}")
    kpi2.metric(label="Churn Rate", value=f"{churn_rate:.2f}%")
    kpi3.metric(label="Average Tenure (Months)", value=f"{avg_tenure:.1f}")

    st.markdown("---")

    # --- ROW 2: Interactive Visualizations ---
    st.subheader("🔍 Data Exploration & Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Churn Distribution")
        fig_churn = px.pie(df, names=target_col, title="Overall Churn Ratio", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_churn, use_container_width=True)
        
    with col2:
        if 'InternetService' in df.columns:
            st.write("### Churn by Internet Service Type")
            fig_internet = px.histogram(df, x='InternetService', color=target_col, barmode='group', title="Internet Service vs Churn")
            st.plotly_chart(fig_internet, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if 'MonthlyCharges' in df.columns:
            st.write("### Monthly Charges vs Churn")
            fig_box = px.box(df, x=target_col, y='MonthlyCharges', color=target_col, title="Monthly Charges Distribution")
            st.plotly_chart(fig_box, use_container_width=True)
            
    with col4:
        if 'Contract' in df.columns:
            st.write("### Contract Type Analysis")
            fig_contract = px.histogram(df, x='Contract', color=target_col, barmode='stack', title="Contract Type vs Churn")
            st.plotly_chart(fig_contract, use_container_width=True)

    st.markdown("---")

    # --- ROW 3: Machine Learning Model Insights ---
    st.subheader("🤖 Machine Learning Model Training (Random Forest)")
    
    if st.checkbox("Model Train Karein aur Evaluation Dekhein"):
        with st.spinner("Model train ho raha hai..."):
            # Encoding categorical variables for quick training
            df_ml = pd.get_dummies(df, drop_first=True)
            
            # Re-identifying X and y after dummy encoding
            X = df_ml.drop(columns=[col for col in df_ml.columns if target_col in col])
            # tracking the exact churn column name
            y_col = [col for col in df_ml.columns if target_col in col][0]
            y = df_ml[y_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Results Display
            ml_col1, ml_col2 = st.columns(2)
            
            with ml_col1:
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig_cm, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig_cm)
                
            with ml_col2:
                st.write("#### Classification Report")
                report = classification_report(y_test, preds, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='YlGnBu'))

else:
    st.info("👋 Dashboard start karne ke liye sidebar se apni **Telco Customer Churn** ki CSV file upload karein.")

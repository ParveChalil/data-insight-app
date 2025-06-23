import streamlit as st
import pandas as pd
import uuid
from utils import *

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

# Global dict to store uploaded DataFrames
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}

st.title("📊 Smart Data Analysis Web App")

# Upload multiple files
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

# Load and store data
for file in uploaded_files:
    if file.name not in st.session_state.dataframes:
        df = pd.read_csv(file)
        name = f"df_{uuid.uuid4().hex[:5]}"
        st.session_state.dataframes[name] = df
        st.success(f"Uploaded: {file.name} → Assigned as `{name}`")

# Select a DataFrame
df_keys = list(st.session_state.dataframes.keys())
selected_df = st.selectbox("Select a DataFrame", df_keys)
df = st.session_state.dataframes[selected_df]

# Show schema
st.subheader("📑 Table Schema")
st.dataframe(get_schema(df))

# Select column
selected_col = st.selectbox("Select Column for Analysis", df.columns)

# Show stats
st.subheader("📈 Column Statistics")
st.json(get_stats(df, selected_col))

# Null handling
st.subheader("🧹 Handle Null Values")
null_action = st.selectbox("Replace nulls in selected column with", ["None", "mean", "median", "mode", "max", "drop"])
if st.button("Apply Null Handling"):
    if null_action != "None":
        df = handle_nulls(df, selected_col, null_action)
        st.success(f"Nulls handled using: {null_action}")

# Object encoding
if st.button("🔡 Encode Object Columns"):
    df = encode_objects(df)
    st.success("Object columns transformed.")

# Train-Test Split
st.subheader("🧪 Train-Test Split")
target_col = st.selectbox("Select target column (optional)", ["None"] + list(df.columns))
test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

if st.button("Create Train/Test Split"):
    tgt_col = target_col if target_col != "None" else None
    X_train, X_test, y_train, y_test = sample_data(df, test_size, tgt_col)
    
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    st.success("Train/Test data created.")
    st.write("X_train", X_train.head())
    st.write("y_train", y_train.head())

# Download final dataset
st.subheader("📥 Download Data")
final_df_name = st.selectbox("Select DataFrame to Download", list(st.session_state.dataframes.keys()))
download = st.session_state.dataframes[final_df_name]

st.download_button(
    label="Download as CSV",
    data=download.to_csv(index=False),
    file_name=f"{final_df_name}.csv",
    mime='text/csv'
)
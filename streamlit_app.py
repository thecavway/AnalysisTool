import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openpyxl
import io

# Set Streamlit page configuration
st.set_page_config(page_title="SPC & Quality Analysis Agent", layout="wide")

# Title
st.title(" SPC & Quality Analysis Agent")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

def download_chart(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(
        label=" Download Chart as PNG",
        data=buf,
        file_name=filename,
        mime="image/png"
    )

# If file is uploaded
if uploaded_file:
    try:
        # Get sheet names and require selection
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select sheet to load (required)", xls.sheet_names)
        if sheet_name:
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
            st.success(f"Data loaded successfully from sheet: {sheet_name}")
            st.write("### Data Preview")
            st.dataframe(df.head())
        else:
            st.warning("Please select a sheet to proceed.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Tool selection
    tool = st.selectbox("Choose Analysis Tool", [
        "SPC Individuals Chart",
        "Pareto Chart",
        "Boxplot",
        "Scatterplot",
        "Run Chart",
        "Pivot Table"
    ])

    # Scatterplot with regression option
    if tool == "Scatterplot":
        x_col = st.selectbox("Select X-axis column", df.columns)
        y_col = st.selectbox("Select Y-axis column", df.columns)
        add_fit = st.checkbox("Add fitted line with R annotation")

        if st.button("Generate Scatterplot"):
            # Clean data for regression
            data_xy = df[[x_col, y_col]].dropna()
            data_xy[x_col] = pd.to_numeric(data_xy[x_col], errors='coerce')
            data_xy[y_col] = pd.to_numeric(data_xy[y_col], errors='coerce')
            data_xy = data_xy.dropna()

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=data_xy[x_col], y=data_xy[y_col], ax=ax)

            if add_fit and not data_xy.empty:
                slope, intercept = np.polyfit(data_xy[x_col], data_xy[y_col], 1)
                r_value = np.corrcoef(data_xy[x_col], data_xy[y_col])[0, 1]
                r_squared = r_value ** 2

                x_vals = np.array(ax.get_xlim())
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color='red', linestyle='--', label=f'Fit line (R={r_squared:.3f})')
                ax.legend()

                # Show regression summary below chart
                st.write(f"**Regression Summary:** Slope = {slope:.3f}, Intercept = {intercept:.3f}, R = {r_squared:.3f}")

            ax.set_title("Scatterplot")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
            download_chart(fig, "scatterplot.png")

    # Other tools remain unchanged (SPC, Pareto, Boxplot, Run Chart, Pivot Table)

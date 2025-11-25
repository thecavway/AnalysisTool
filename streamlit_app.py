import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openpyxl
import io

st.set_page_config(page_title="SPC & Quality Analysis Agent", layout="wide")
st.title(" SPC & Quality Analysis Agent")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

def download_chart(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(label=" Download Chart as PNG", data=buf, file_name=filename, mime="image/png")

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select sheet to load (required)", xls.sheet_names)
        if not sheet_name:
            st.warning("Please select a sheet to proceed.")
            st.stop()

        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
        st.success(f"Data loaded successfully from sheet: {sheet_name}")
        st.write("### Data Preview")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    tool = st.selectbox("Choose Analysis Tool", [
        "SPC Individuals Chart",
        "Pareto Chart",
        "Boxplot",
        "Scatterplot",
        "Run Chart",
        "Pivot Table",
        "Capability Index Analysis"
    ])

    # SPC Individuals Chart
    if tool == "SPC Individuals Chart":
        col = st.selectbox("Select column for SPC chart", df.columns)
        data = df[col].dropna().tolist()

        st.write("### Select subset for initial control limits")
        subset_option = st.radio("Choose subset method", ["All data", "Select range"])

        if subset_option == "Select range":
            start_idx = st.number_input("Start index (0-based)", min_value=0, max_value=len(data)-1, value=0)
            end_idx = st.number_input("End index (0-based)", min_value=start_idx, max_value=len(data)-1, value=len(data)-1)
            subset_data = data[start_idx:end_idx+1]
        else:
            subset_data = data

        recalc_points_input = st.text_input("Enter recalculation points (comma-separated indices)", "")
        recalc_points = sorted([int(x) for x in recalc_points_input.split(",") if x.strip().isdigit()])

        if st.button("Generate SPC Chart"):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data, marker='o', linestyle='-', color='black')

            split_points = [0] + recalc_points + [len(data)]

            for i in range(len(split_points)-1):
                seg_start = split_points[i]
                seg_end = split_points[i+1]
                segment = data[seg_start:seg_end]

                mean_val = pd.Series(segment).mean()
                std_val = pd.Series(segment).std()

                ax.hlines(mean_val, seg_start, seg_end-1, colors='blue', linestyles='--')
                ax.hlines(mean_val + 3 * std_val, seg_start, seg_end-1, colors='red', linestyles=':')
                ax.hlines(mean_val - 3 * std_val, seg_start, seg_end-1, colors='red', linestyles=':')

            ax.set_title("SPC Chart with Split Control Limits")
            ax.set_xlabel("Observation")
            ax.set_ylabel(col)
            st.pyplot(fig)
            download_chart(fig, "spc_chart.png")

    # Pareto Chart
    elif tool == "Pareto Chart":
        category_col = st.selectbox("Select Category Column", df.columns)
        freq_col = st.selectbox("Select Frequency Column (optional)", ["None"] + list(df.columns))

        if st.button("Generate Pareto Chart"):
            if freq_col != "None":
                pareto_df = df.groupby(category_col)[freq_col].sum().sort_values(ascending=False)
            else:
                pareto_df = df[category_col].value_counts()

            cumulative = pareto_df.cumsum() / pareto_df.sum() * 100

            fig, ax1 = plt.subplots(figsize=(10, 6))
            pareto_df.plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_ylabel('Frequency')
            ax2 = ax1.twinx()
            ax2.plot(cumulative.values, color='red', marker='D', linestyle='-')
            ax2.set_ylabel('Cumulative %')
            ax1.set_title("Pareto Chart")
            ax2.axhline(80, color='green', linestyle='--')
            st.pyplot(fig)
            download_chart(fig, "pareto_chart.png")

    # Boxplot
    elif tool == "Boxplot":
        measure_col = st.selectbox("Select measure column (numeric)", df.columns)
        category_col = st.selectbox("Select category column (optional)", ["None"] + list(df.columns))
        outlier_option = st.radio("Outlier display option", ["Show outliers", "Hide outliers", "Highlight outliers"])

        if st.button("Generate Boxplot"):
            fig, ax = plt.subplots(figsize=(10, 6))

            if category_col != "None":
                sns.boxplot(x=df[category_col], y=df[measure_col], ax=ax, showfliers=(outlier_option != "Hide outliers"))
            else:
                sns.boxplot(y=df[measure_col], ax=ax, showfliers=(outlier_option != "Hide outliers"))

            outlier_data = None
            if outlier_option == "Highlight outliers":
                q1 = df[measure_col].quantile(0.25)
                q3 = df[measure_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_data = df[(df[measure_col] < lower_bound) | (df[measure_col] > upper_bound)]

                if category_col != "None":
                    sns.scatterplot(x=outlier_data[category_col], y=outlier_data[measure_col], color='red', ax=ax, label='Outliers')
                else:
                    sns.scatterplot(y=outlier_data[measure_col], color='red', ax=ax, label='Outliers')

            ax.set_title("Boxplot")
            ax.set_ylabel(measure_col)
            if category_col != "None":
                ax.set_xlabel(category_col)
            st.pyplot(fig)
            download_chart(fig, "boxplot.png")

            if outlier_data is not None and not outlier_data.empty:
                outlier_csv = outlier_data.to_csv(index=False).encode('utf-8')
                st.download_button(label=" Download Outliers as CSV", data=outlier_csv, file_name="outliers.csv", mime="text/csv")

    # Scatterplot with regression
    elif tool == "Scatterplot":
        x_col = st.selectbox("Select X-axis column", df.columns)
        y_col = st.selectbox("Select Y-axis column", df.columns)
        add_fit = st.checkbox("Add fitted line with R annotation")

        if st.button("Generate Scatterplot"):
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

                st.write(f"**Regression Summary:** Slope = {slope:.3f}, Intercept = {intercept:.3f}, R = {r_squared:.3f}")

            ax.set_title("Scatterplot")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig)
            download_chart(fig, "scatterplot.png")

    # Run Chart
    elif tool == "Run Chart":
        time_col = st.selectbox("Select Time column (optional)", ["None"] + list(df.columns))
        value_col = st.selectbox("Select Value column", df.columns)

        if st.button("Generate Run Chart"):
            fig, ax = plt.subplots(figsize=(10, 6))

            if time_col != "None":
                ax.plot(df[time_col], df[value_col], marker='o')
                ax.set_xlabel(time_col)
            else:
                ax.plot(df[value_col].reset_index(drop=True), marker='o')
                ax.set_xlabel("Observation")

            median_val = df[value_col].median()
            ax.axhline(median_val, color='red', linestyle='--')
            ax.set_title("Run Chart with Median Line")
            ax.set_ylabel(value_col)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            download_chart(fig, "run_chart.png")

    # Pivot Table with safeguards
    elif tool == "Pivot Table":
        index_cols = st.multiselect("Select index columns", df.columns)
        column_cols = st.multiselect("Select columns (optional)", df.columns)
        value_col = st.selectbox("Select value column", df.columns)
        aggfunc = st.selectbox("Select aggregation function", ["mean", "sum", "count"])

        if st.button("Generate Pivot Table"):
            try:
                # Remove duplicates between index and columns
                index_cols = [col for col in index_cols if col not in column_cols]

                # Convert non-scalar values to strings
                for col in index_cols + column_cols:
                    if not pd.api.types.is_scalar(df[col].iloc[0]):
                        df[col] = df[col].astype(str)

                pivot_df = pd.pivot_table(df,
                                           index=index_cols if index_cols else None,
                                           columns=column_cols if column_cols else None,
                                           values=value_col,
                                           aggfunc=aggfunc)

                st.write("### Pivot Table")
                st.dataframe(pivot_df)

                pivot_csv = pivot_df.to_csv().encode('utf-8')
                st.download_button(label=" Download Pivot Table as CSV", data=pivot_csv,
                                   file_name="pivot_table.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error generating pivot table: {e}")

    # Capability Index Analysis with single-sided tolerance
    elif tool == "Capability Index Analysis":
        col = st.selectbox("Select numeric column for capability analysis", df.columns)
        lsl = st.text_input("Enter Lower Spec Limit (optional)", "")
        usl = st.text_input("Enter Upper Spec Limit (optional)", "")
        index_type = st.radio("Select index type", ["Cp", "Cpk", "Ppk"])

        if st.button("Calculate Capability Index"):
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            mean_val = data.mean()
            std_within = data.std(ddof=1)
            std_overall = data.std(ddof=0)

            lsl_val = float(lsl) if lsl.strip() != "" else None
            usl_val = float(usl) if usl.strip() != "" else None

            cp = None
            if lsl_val is not None and usl_val is not None and std_within > 0:
                cp = (usl_val - lsl_val) / (6 * std_within)

            cpk = None
            if std_within > 0:
                vals = []
                if usl_val is not None:
                    vals.append((usl_val - mean_val) / (3 * std_within))
                if lsl_val is not None:
                    vals.append((mean_val - lsl_val) / (3 * std_within))
                if vals:
                    cpk = min(vals)

            ppk = None
            if std_overall > 0:
                vals = []
                if usl_val is not None:
                    vals.append((usl_val - mean_val) / (3 * std_overall))
                if lsl_val is not None:
                    vals.append((mean_val - lsl_val) / (3 * std_overall))
                if vals:
                    ppk = min(vals)

            if index_type == "Cp":
                index_value = cp if cp is not None else 0
            elif index_type == "Cpk":
                index_value = cpk if cpk is not None else 0
            else:
                index_value = ppk if ppk is not None else 0

            capability_score = min(index_value / 1.33 * 100, 10000) if index_value > 0 else 0

            st.write(f"**{index_type} = {index_value:.3f}**")
            st.metric(label="Capability Score (%)", value=f"{capability_score:.2f}%")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data, bins=20, ax=ax, color='skyblue')
            if lsl_val is not None:
                ax.axvline(lsl_val, color='green', linestyle='--', label=f'LSL = {lsl_val}')
            if usl_val is not None:
                ax.axvline(usl_val, color='red', linestyle='--', label=f'USL = {usl_val}')
            ax.axvline(mean_val, color='black', linestyle='-', label=f'Mean = {mean_val:.2f}')
            ax.set_title("Capability Index Analysis")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
            download_chart(fig, "capability_index.png")

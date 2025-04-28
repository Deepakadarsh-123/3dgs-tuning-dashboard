import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

st.set_page_config(layout="wide")
st.title("📌 Postshot Parameter Analysis for Rebar Detection and Insights")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your postshot trial data Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, header=1)
    df.columns = [str(c).strip() for c in df.columns]

    # Clean and safely convert "Downsampled to" to float
    def clean_downsample(x):
        x = str(x).lower().replace('px', '').replace('k', '000')
        return pd.to_numeric(x, errors='coerce')

    if "Downsampled to" in df.columns:
        df["Downsampled to"] = df["Downsampled to"].apply(clean_downsample)

    # Clean percentages and times
    if "Avg Mean % Abs. error" in df.columns:
        df["Avg Mean % Abs. error"] = df["Avg Mean % Abs. error"].astype(str).str.replace('%', '').astype(float)

    def time_to_minutes(t):
        if isinstance(t, str) and ":" in t:
            parts = list(map(int, t.split(":")))
            return parts[0]*60 + parts[1] + parts[2]/60 if len(parts) == 3 else None
        return None

    if "Time taken in training" in df.columns:
        df["Training Time (min)"] = df["Time taken in training"].apply(time_to_minutes)

    st.dataframe(df)

    st.header("📊 Customizable Scatter and Correlation Explorer")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    x_axis = st.selectbox("Select X-axis", numeric_cols)
    y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1)
    color = st.selectbox("Color by", [None] + numeric_cols)

    if x_axis and y_axis:
        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            color=color if color else None,
            trendline="ols",
            title=f"Scatter plot: {x_axis} vs {y_axis}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show correlation coefficient
        if not df[[x_axis, y_axis]].dropna().empty:
            corr_value = df[x_axis].corr(df[y_axis])
            st.metric(label=f"Correlation between {x_axis} and {y_axis}", value=f"{corr_value:.2f}")

    st.header("📈 Advanced Visualization")

    # Pairwise Scatter Plot Matrix
    st.subheader("📊 Pairwise Scatter Plot Matrix")
    selected_cols_matrix = st.multiselect("Select dimensions for scatter matrix", numeric_cols, default=numeric_cols[:5])
    if selected_cols_matrix:
        fig_matrix = px.scatter_matrix(df, dimensions=selected_cols_matrix, color=color if color else None)
        st.plotly_chart(fig_matrix, use_container_width=True)

    # Parallel Coordinates Plot
    st.subheader("🧭 Parallel Coordinates Plot")
    selected_cols_parallel = st.multiselect("Select dimensions for parallel plot", numeric_cols, default=numeric_cols[:5], key='parallel')
    if selected_cols_parallel:
        fig_parallel = px.parallel_coordinates(df, dimensions=selected_cols_parallel, color=df[color] if color else df[selected_cols_parallel[0]])
        st.plotly_chart(fig_parallel, use_container_width=True)

    # Heatmap: SSIM vs Training Steps vs Splats
    st.subheader("🔥 Heatmap: SSIM vs Training Steps vs Splats")
    heatmap_x = st.selectbox("Heatmap X-axis", numeric_cols, index=numeric_cols.index("Training Steps (k)"))
    heatmap_y = st.selectbox("Heatmap Y-axis", numeric_cols, index=numeric_cols.index("Max Splats counts (in k)"))
    if heatmap_x and heatmap_y and "SSIM" in df.columns:
        pivot = df.pivot_table(values="SSIM", index=heatmap_y, columns=heatmap_x)
        fig_heatmap = px.imshow(pivot, text_auto=True, title="Heatmap of SSIM")
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # SSIM vs Training Time Line Plot
    st.subheader("📈 SSIM vs Training Time")
    if "Training Time (min)" in df.columns and "SSIM" in df.columns:
        fig_ssim_time = px.line(df, x="Training Time (min)", y="SSIM", markers=True, title="SSIM vs Training Time")
        st.plotly_chart(fig_ssim_time, use_container_width=True)

    # SECTION: Summary Highlights
    st.subheader("✅ Best Trials (Lowest Avg Mean Abs. Error)")
    if "Avg Mean Abs. error (in mm)" in df.columns:
        best_trials = df.sort_values("Avg Mean Abs. error (in mm)").head(5)
        st.dataframe(best_trials[["Trial", "Avg Mean Abs. error (in mm)", "Nos. of splats generated (in k)", "SSIM", "Training Steps (k)"]])

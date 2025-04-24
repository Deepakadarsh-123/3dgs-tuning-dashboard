import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide")
st.title("🔍 Multi-Parameter Tuning Dashboard for 3DGS Trials")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your trial data Excel file", type=["xlsx", "xls", "csv"])

if uploaded_file:
    # Read data
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Data successfully loaded!")
    st.dataframe(df)

    # Convert time to minutes
    def time_to_minutes(t):
        if isinstance(t, str) and ":" in t:
            parts = list(map(int, t.split(":")))
            return parts[0]*60 + parts[1] + parts[2]/60 if len(parts) == 3 else None
        return None

    df["Training Time (min)"] = df["Time taken in training"].apply(time_to_minutes)

    # Scatter Plot Matrix (pairplot style)
    st.subheader("📊 Pairwise Scatter Plot Matrix")
    metrics = ["Downsampled to", "Nos. of best photos", "Training Steps (k)", "Max Splats counts (in k)", "Training Time (min)", "SSIM"]
    fig1 = px.scatter_matrix(df, dimensions=metrics, color="SSIM", title="Scatter Matrix of Tuning Parameters and Output")
    st.plotly_chart(fig1, use_container_width=True)

    # Parallel Coordinates Plot
    st.subheader("🧭 Parallel Coordinates Plot")
    fig2 = px.parallel_coordinates(df, 
        dimensions=["Downsampled to", "Nos. of best photos", "Training Steps (k)", "Max Splats counts (in k)", "Training Time (min)", "SSIM"],
        color="SSIM",
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Parallel Coordinates for Parameter Exploration"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Heatmap Grid - SSIM vs Training Steps and Splats
    st.subheader("🔥 Heatmap: SSIM vs Training Steps vs Splats")
    pivot = df.pivot_table(values="SSIM", index="Training Steps (k)", columns="Max Splats counts (in k)")
    fig3 = px.imshow(pivot, text_auto=True, title="SSIM Heatmap by Training Steps and Max Splats")
    st.plotly_chart(fig3, use_container_width=True)

    # SSIM vs Time Line Plot
    st.subheader("📈 SSIM vs Training Time")
    fig4 = px.line(df, x="Training Time (min)", y="SSIM", markers=True, text="Trial", title="SSIM Progression Over Training Time")
    st.plotly_chart(fig4, use_container_width=True)

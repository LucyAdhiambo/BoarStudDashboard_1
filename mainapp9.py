import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# -----------------------------------
# Streamlit page setup
# -----------------------------------
st.set_page_config(page_title="🐗 Boar Semen Analysis Dashboard", layout="wide")
st.title("🐗 Weekly Automated Boar Semen Analysis Dashboard")

st.markdown("""
This dashboard lets you upload weekly semen data, calculate averages, 
compare breeds, view all graphs, and adjust views in real time.
""")

# -----------------------------------
# File upload section
# -----------------------------------
uploaded = st.file_uploader("📤 Upload your Excel file", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)
    st.success(f"✅ File uploaded successfully: {uploaded.name}")

    # -----------------------------------
    # Clean data
    # -----------------------------------
    df.replace("-", np.nan, inplace=True)
    amount_cols = ["Amount", "Amount.1", "Amount.2", "Amount.3"]
    tubes_cols = ["Tubes ", "Tubes .1", "Tubes .2", "Tubes .3"]

    for col in amount_cols + tubes_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate averages
    df["Avg_Amount"] = df[amount_cols].mean(axis=1)
    df["Avg_Tubes"] = df[tubes_cols].mean(axis=1)
    df["DOSES"] = df["Avg_Tubes"] / 2

    # -----------------------------------
    # Filters and Controls
    # -----------------------------------
    st.sidebar.header("🔍 Filter & Graph Controls")

    breeds = df["Breed"].unique().tolist()
    selected_breeds = st.sidebar.multiselect("Select Breeds to Display", options=breeds, default=breeds)

    chart_type = st.sidebar.radio("Select Chart Type", ["Bar Chart", "Trend Line"])

    metric = st.sidebar.selectbox(
        "Select Metric to Visualize",
        ["Avg_Amount", "Avg_Tubes", "DOSES"]
    )

    # Filter data by breed
    df_filtered = df[df["Breed"].isin(selected_breeds)]

    st.subheader("📄 Data Preview")
    st.dataframe(df_filtered.head())

    # -----------------------------------
    # Summary Calculations
    # -----------------------------------
    best_boars = (
        df_filtered.loc[df_filtered.groupby("Breed")[metric].idxmax(), ["Breed", "Boar No", metric]]
        .reset_index(drop=True)
        .rename(columns={metric: f"Best_Boar_{metric}"})
    )

    breed_avg = (
        df_filtered.groupby("Breed")[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"Breed_Avg_{metric}"})
    )

    summary = pd.merge(breed_avg, best_boars, on="Breed")
    st.subheader("📊 Summary Statistics")
    st.dataframe(summary)

    # -----------------------------------
    # Individual Boar Performance
    # -----------------------------------
    st.subheader("🐗 Individual Boar Performance (DOSES)")
    boar_performance = df_filtered[["Boar No", "Breed", "DOSES"]].sort_values(by=["Breed", "Boar No"])
    st.dataframe(boar_performance)

    st.subheader("📊 DOSES per Boar")
    fig, ax = plt.subplots(figsize=(10, max(6, len(boar_performance)*0.3)))  # dynamic height
    sns.barplot(
        data=boar_performance,
        y="Boar No",  # horizontal bars
        x="DOSES",
        hue="Breed",
        palette={2: 'blue', 3: 'yellow', 800: 'green'},
        ax=ax
    )
    plt.title("Individual Boar Performance (DOSES)")
    plt.xlabel("DOSES")
    plt.ylabel("Boar No")
    plt.tight_layout()

    # Add exact DOSES value at the end of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    # Fix legend labels
    handles, _ = ax.get_legend_handles_labels()
    legend_labels = {2: '2-LR', 3: '3-LW', 800: '800-DUROC'}
    ax.legend(handles, [legend_labels.get(int(h.get_label()), h.get_label()) for h in handles], title="Breed")

    st.pyplot(fig)

    # -----------------------------------
    # Visualization Section
    # -----------------------------------
    st.subheader("📈 Visualizations")

    # Define colors and legend labels
    breed_colors = {2: 'blue', 3: 'yellow', 800: 'green'}

    if chart_type == "Bar Chart":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=breed_avg, 
            x="Breed", 
            y=f"Breed_Avg_{metric}", 
            hue="Breed", 
            palette=breed_colors, 
            ax=ax
        )
        plt.title(f"Average {metric} per Breed")
        plt.ylabel(f"Average {metric}")
        plt.xlabel("Breed")
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        # Fix legend labels
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, [legend_labels.get(int(h.get_label()), h.get_label()) for h in handles], title="Breed")
        st.pyplot(fig)

    elif chart_type == "Trend Line":
        # Select columns related to the chosen metric
        value_cols = [col for col in df.columns if metric.split('_')[1] in col]

        # Melt the filtered DataFrame safely
        melted = df_filtered.melt(
            id_vars=["Boar No", "Breed"], 
            value_vars=value_cols,
            var_name="Collection", 
            value_name="Value"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=melted, 
            x="Collection", 
            y="Value",
            hue="Breed",
            marker="o",
            palette=breed_colors,
            ax=ax
        )
        plt.title(f"{metric} Trend Across Collections per Breed")
        plt.ylabel(metric)
        plt.xlabel("Collection Number")
        # Fix legend labels
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, [legend_labels.get(int(h.get_label()), h.get_label()) for h in handles], title="Breed")
        st.pyplot(fig)

    # -----------------------------------
    # Export Results
    # -----------------------------------
    st.subheader("💾 Export Results")
    if st.button("Export to Excel"):
        os.makedirs("reports", exist_ok=True)
        output_path = f"reports/semen_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Raw_Data")
            summary.to_excel(writer, index=False, sheet_name="Summary")
        st.success(f"✅ Exported to {output_path}")

else:
    st.info("👈 Please upload your weekly Excel file to start the analysis.")

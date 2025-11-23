from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app.utils.data_utils import (PROCESSED_DATA_PATH, RESULTS_PATH,
                                  get_feature_columns, get_test_data,
                                  get_train_data, make_pie)

st.set_page_config(layout="wide")

st.title("ID Fan Failure Prediction System")
st.markdown("---")

col1, _, _ = st.columns([2, 1, 1])

with col1:
    st.markdown(
        """
    ### Welcome to the ID Fan Failure Prediction System
    
    This application uses traditional machine learning models to predict IDfan failures.
    By providing 21 key operational features, the system can identify early 
    warning signs of potential failures, enabling proactive maintenance and reducing downtime.
    """
    )

st.markdown("---")
# ------------------------

st.subheader("Project Overview")
tab1, tab2 = st.tabs(["Dataset Information", "Model Performance"])

with tab1:
    if PROCESSED_DATA_PATH.exists():
        try:
            df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", f"{len(df):,}")

            with col2:
                if "Target_Class" in df.columns:
                    normal_count = (df["Target_Class"] == 0).sum()
                    failure_count = (df["Target_Class"] == 1).sum()
                    st.metric("Normal Operations", f"{normal_count:,}")

            with col3:
                if "Target_Class" in df.columns:
                    st.metric("Towards Failure", f"{failure_count:,}")

            with col4:
                features_df = get_feature_columns(df)
                st.metric("Features", f"{len(features_df.columns)}")
            # Train/Test Distribution
            try:
                # Load data first
                X_train, y_train = get_train_data()
                X_test, y_test = get_test_data()

                # Combine features with target for display
                train_df = X_train.copy()
                train_df["Target_Class"] = y_train.values
                train_df["Target_Label"] = train_df["Target_Class"].map(
                    {0: "Normal", 1: "Towards Failure"}
                )

                test_df = X_test.copy()
                test_df["Target_Class"] = y_test.values
                test_df["Target_Label"] = test_df["Target_Class"].map(
                    {0: "Normal", 1: "Towards Failure"}
                )

                # Train/Test Dataframes in expandable sections
                st.markdown("#### Train/Test Data")
                train, test = st.columns(2)
                with train:
                    with st.expander(
                        f"Train Dataset ({len(train_df):,} records)", expanded=False
                    ):
                        st.dataframe(train_df, width="stretch", height=400)
                        st.caption(
                            f"Train dataset contains {len(train_df):,} records with {len(X_train.columns)} features."
                        )

                with test:
                    with st.expander(
                        f"Test Dataset ({len(test_df):,} records)", expanded=False
                    ):
                        st.dataframe(test_df, width="stretch", height=400)
                        st.caption(
                            f"Test dataset contains {len(test_df):,} records with {len(X_test.columns)} features."
                        )

                train_counts = y_train.value_counts()
                test_counts = y_test.value_counts()

                st.markdown("#### Train/Test Distribution")
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))

                make_pie(axs[0], train_counts, "Train Distribution")
                make_pie(axs[1], test_counts, "Test Distribution")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info(
                    "Train/Test distribution will be displayed once data is loaded."
                )

        except Exception as e:
            st.info(
                "Dataset information will be displayed here once data is processed."
            )

with tab2:
    if RESULTS_PATH.exists():
        try:
            results_df = pd.read_csv(RESULTS_PATH)
            results_df = results_df.sort_values(by="F1 Score", ascending=False)

            st.markdown("#### Model Performance Comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(
                    results_df[
                        ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
                    ],
                    width="stretch",
                    hide_index=True,
                )

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
                x = range(len(results_df))
                width = 0.2

                for i, metric in enumerate(metrics):
                    offset = (i - 1.5) * width
                    ax.bar(
                        [xi + offset for xi in x],
                        results_df[metric],
                        width,
                        label=metric,
                        alpha=0.8,
                    )

                ax.set_xlabel("Models", fontweight="bold")
                ax.set_ylabel("Score", fontweight="bold")
                ax.set_title(
                    "Model Performance Metrics", fontweight="bold", fontsize=14
                )
                ax.set_xticks(x)
                ax.set_xticklabels(results_df["Model"], rotation=45, ha="right")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        except Exception as e:
            st.info(
                "Model performance metrics will be displayed here once models are trained."
            )

st.markdown("---")

st.markdown(
    """
<div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); border-radius: 12px; box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3); margin: 30px 0;'>
    <h3 style='color: white; margin-bottom: 15px; font-size: 24px;'>Ready to Get Started?</h3>
    <p style='color: #e0f2fe; font-size: 16px; margin: 0;'>Navigate to the <strong style='color: white;'>Predictions</strong> page to begin making predictions or evaluating our models.</p>
</div>
""",
    unsafe_allow_html=True,
)

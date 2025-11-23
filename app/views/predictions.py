from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import metrics as skm

from app.tools.model_evaluator import ModelEvaluator
from app.utils.data_utils import (get_feature_column_names_list,
                                  get_feature_columns)

st.set_page_config(layout="wide", page_title="Model Evaluation")

PROJ_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = PROJ_ROOT / "output" / "ckpts"
OUTPUT_PATH = PROJ_ROOT / "output"
TEST_DATASET_PATH = OUTPUT_PATH / "test_dataset.csv"

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "SVM (RBF)",
    "KNN",
    "MLP (Neural Net)",
    "XGBoost",
    "LightGBM",
    "Extra Trees",
]


@st.cache_data
def load_scaler():
    scaler_path = CKPT_PATH / "scaler.joblib"
    if not scaler_path.exists():
        return None
    return joblib.load(scaler_path)


@st.cache_data
def load_model(model_name):
    model_filename = model_name.replace(" ", "_") + ".joblib"
    model_path = CKPT_PATH / model_filename
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_test_dataset():
    if not TEST_DATASET_PATH.exists():
        return None
    try:
        df = pd.read_csv(TEST_DATASET_PATH, index_col=0, parse_dates=True)
        return df
    except Exception:
        return None


def validate_features(df, feature_columns):
    missing_cols = set(feature_columns) - set(df.columns)
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return False

    if df[feature_columns].isnull().any().any():
        st.error(
            "Data contains missing values. Please handle missing values before prediction."
        )
        return False

    return True


def prepare_features(df, feature_columns):
    return df[feature_columns].copy()


# SESSION STATE
if "data" not in st.session_state:
    st.session_state.data = None
if "y_true" not in st.session_state:
    st.session_state.y_true = None
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "all_predictions" not in st.session_state:
    st.session_state.all_predictions = {}
if "evaluation_mode" not in st.session_state:
    st.session_state.evaluation_mode = "All Models"

# --- MAIN UI ---

st.title("Model Predictions")
st.markdown(
    "Make predictions using trained machine learning models or evaluate models on the test dataset."
)

# Get feature columns list
FEATURE_COLUMNS = get_feature_column_names_list()
if FEATURE_COLUMNS is None:
    st.error("Unable to load feature columns. Please ensure the processed data exists.")
    st.stop()

# --- SECTION 1: DATA INPUT ---
with st.container(border=True):
    st.subheader("Data Input")

    col_opt, col_act = st.columns([1, 2])

    with col_opt:
        input_method = st.radio(
            "Source:",
            ["Upload CSV File", "Load Test Dataset"],
            horizontal=True,
            label_visibility="collapsed",
        )

    with col_act:
        if input_method == "Upload CSV File":
            col_dl, col_up = st.columns(2)
            with col_dl:
                # Generate template
                template_df = pd.DataFrame(columns=FEATURE_COLUMNS)
                template_csv = template_df.to_csv(index=False)
                st.download_button(
                    label="Download Template CSV",
                    data=template_csv,
                    file_name="prediction_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_up:
                uploaded_file = st.file_uploader(
                    "Upload CSV", type=["csv"], label_visibility="collapsed"
                )

            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    if validate_features(data, FEATURE_COLUMNS):
                        st.session_state.data = data
                        st.session_state.y_true = None  # Reset
                        st.session_state.X_scaled = None  # Reset
                        st.session_state.all_predictions = {}  # Reset
                        st.success(f"Loaded {len(data)} rows.")
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")

        elif input_method == "Load Test Dataset":
            load_btn = st.button(
                "Load Default Test Dataset", type="primary", use_container_width=True
            )

            if load_btn:
                with st.spinner("Loading..."):
                    data = load_test_dataset()
                    if data is not None:
                        y_true = None
                        if "Target_Class" in data.columns:
                            y_true = data["Target_Class"].values

                        if validate_features(data, FEATURE_COLUMNS):
                            st.session_state.data = data
                            st.session_state.y_true = y_true
                            st.session_state.X_scaled = None
                            st.session_state.all_predictions = {}
                            st.success(f"Loaded test dataset ({len(data)} rows).")
                    else:
                        st.error("Test dataset file not found.")

    # Show Data Preview if loaded
    if st.session_state.data is not None:
        st.dataframe(st.session_state.data[FEATURE_COLUMNS].head(), width="stretch")


# --- SECTION 2: EVALUATION ---
if st.session_state.data is not None:
    scaler = load_scaler()
    if scaler is None:
        st.error("Scaler missing. Run training first.")
    else:
        # Scale Data Once
        if st.session_state.X_scaled is None:
            X = prepare_features(st.session_state.data, FEATURE_COLUMNS)
            st.session_state.X_scaled = scaler.transform(X)

        st.markdown("---")
        with st.container(border=True):
            st.subheader("Run Evaluation")

            # Mode Selection
            if st.session_state.y_true is not None:
                st.session_state.evaluation_mode = st.radio(
                    "Mode",
                    ["All Models", "Single Model"],
                    horizontal=True,
                    label_visibility="collapsed",
                )
            else:
                st.session_state.evaluation_mode = "Single Model"

            st.markdown("")  # Spacer

            # --- ALL MODELS MODE ---
            if (
                st.session_state.evaluation_mode == "All Models"
                and st.session_state.y_true is not None
            ):
                run_all_btn = st.button(
                    "Evaluate All Models", type="primary", use_container_width=True
                )

                if run_all_btn:
                    with st.spinner("Processing all models..."):
                        progress_bar = st.progress(0)
                        for idx, model_name in enumerate(MODEL_NAMES):
                            model = load_model(model_name)
                            if model is not None:
                                y_pred_proba = model.predict_proba(
                                    st.session_state.X_scaled
                                )[:, 1]
                                y_pred = model.predict(st.session_state.X_scaled)
                                st.session_state.all_predictions[model_name] = {
                                    "y_pred": y_pred,
                                    "y_pred_proba": y_pred_proba,
                                }
                            progress_bar.progress((idx + 1) / len(MODEL_NAMES))
                        st.success("Evaluation Complete.")

                # Display Results
                if st.session_state.all_predictions:
                    # Metrics Calculation
                    metrics_data = []
                    for (
                        model_name,
                        pred_data,
                    ) in st.session_state.all_predictions.items():
                        y_pred = pred_data["y_pred"]
                        y_pred_proba = pred_data["y_pred_proba"]

                        metrics_data.append(
                            {
                                "Model": model_name,
                                "Accuracy": skm.accuracy_score(
                                    st.session_state.y_true, y_pred
                                ),
                                "Precision": skm.precision_score(
                                    st.session_state.y_true, y_pred, zero_division=0
                                ),
                                "Recall": skm.recall_score(
                                    st.session_state.y_true, y_pred, zero_division=0
                                ),
                                "F1 Score": skm.f1_score(
                                    st.session_state.y_true, y_pred, zero_division=0
                                ),
                                "ROC AUC": skm.roc_auc_score(
                                    st.session_state.y_true, y_pred_proba
                                ),
                                "PR AUC": skm.average_precision_score(
                                    st.session_state.y_true, y_pred_proba
                                ),
                            }
                        )
                    metrics_df = pd.DataFrame(metrics_data)

                    # TABS FOR VISUALIZATION
                    tab1, tab2, tab3, tab4 = st.tabs(
                        [
                            "Metrics Data",
                            "Confusion Matrices",
                            "Class Reports",
                            "Curves",
                        ]
                    )

                    with tab1:
                        # Format only numeric columns
                        format_dict = {
                            col: "{:.3f}"
                            for col in metrics_df.columns
                            if col != "Model"
                        }
                        st.dataframe(
                            metrics_df.style.format(format_dict),
                            width="stretch",
                            hide_index=True,
                        )

                    with tab2:
                        num_models = len(st.session_state.all_predictions)
                        cols = 4
                        rows = (num_models + cols - 1) // cols
                        fig_cm, axes_cm = plt.subplots(
                            rows, cols, figsize=(20, 5 * rows)
                        )
                        axes_cm = axes_cm.flatten()

                        for idx, (model_name, pred_data) in enumerate(
                            st.session_state.all_predictions.items()
                        ):
                            cm = skm.confusion_matrix(
                                st.session_state.y_true, pred_data["y_pred"]
                            )
                            sns.heatmap(
                                cm,
                                annot=True,
                                fmt="d",
                                cmap="Blues",
                                ax=axes_cm[idx],
                                cbar=False,
                            )
                            axes_cm[idx].set_title(model_name)

                        # Hide empty subplots (if any)
                        for i in range(idx + 1, len(axes_cm)):
                            axes_cm[i].axis("off")

                        plt.tight_layout()
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                    with tab3:
                        # Classification report visualization
                        fig_rep, axes_rep = plt.subplots(
                            rows, cols, figsize=(22, 5 * rows)
                        )
                        axes_rep = axes_rep.flatten()
                        for idx, (model_name, pred_data) in enumerate(
                            st.session_state.all_predictions.items()
                        ):
                            report = skm.classification_report(
                                st.session_state.y_true,
                                pred_data["y_pred"],
                                output_dict=True,
                            )
                            sns.heatmap(
                                pd.DataFrame(report).transpose().iloc[:-1, :3],
                                annot=True,
                                cmap="Greens",
                                ax=axes_rep[idx],
                                cbar=False,
                            )
                            axes_rep[idx].set_title(model_name)

                        for i in range(idx + 1, len(axes_rep)):
                            axes_rep[i].axis("off")
                        plt.tight_layout()
                        st.pyplot(fig_rep)
                        plt.close(fig_rep)

                    with tab4:
                        # Split Curves side by side
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("ROC Curves")
                            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                            for (
                                m_name,
                                p_data,
                            ) in st.session_state.all_predictions.items():
                                fpr, tpr, _ = skm.roc_curve(
                                    st.session_state.y_true, p_data["y_pred_proba"]
                                )
                                auc = skm.auc(fpr, tpr)
                                ax_roc.plot(fpr, tpr, label=f"{m_name} ({auc:.2f})")
                            ax_roc.plot([0, 1], [0, 1], "k--")
                            ax_roc.legend()
                            st.pyplot(fig_roc)
                            plt.close(fig_roc)

                        with c2:
                            st.caption("Precision-Recall Curves")
                            fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
                            for (
                                m_name,
                                p_data,
                            ) in st.session_state.all_predictions.items():
                                prec, rec, _ = skm.precision_recall_curve(
                                    st.session_state.y_true, p_data["y_pred_proba"]
                                )
                                avg_pr = skm.average_precision_score(
                                    st.session_state.y_true, p_data["y_pred_proba"]
                                )
                                ax_pr.plot(rec, prec, label=f"{m_name} ({avg_pr:.2f})")
                            ax_pr.legend()
                            st.pyplot(fig_pr)
                            plt.close(fig_pr)

            # --- SINGLE MODEL MODE ---
            else:
                col_sel, col_btn = st.columns([3, 1])
                with col_sel:
                    model_selection = st.selectbox(
                        "Select Model", MODEL_NAMES, label_visibility="collapsed"
                    )
                with col_btn:
                    run_pred_btn = st.button(
                        "Predict", type="primary", use_container_width=True
                    )

                if run_pred_btn:
                    model = load_model(model_selection)
                    if model is not None:
                        y_pred_proba = model.predict_proba(st.session_state.X_scaled)[
                            :, 1
                        ]
                        y_pred = model.predict(st.session_state.X_scaled)
                        st.session_state.all_predictions[model_selection] = {
                            "y_pred": y_pred,
                            "y_pred_proba": y_pred_proba,
                        }

                # Display Results
                if model_selection in st.session_state.all_predictions:
                    pred_data = st.session_state.all_predictions[model_selection]
                    y_pred = pred_data["y_pred"]
                    y_pred_proba = pred_data["y_pred_proba"]

                    # Result Dataframe
                    results_df = pd.DataFrame(
                        {
                            "Prediction": y_pred,
                            "Prob(Failure)": y_pred_proba,
                            "Label": [
                                "Towards Failure" if p == 1 else "Normal"
                                for p in y_pred
                            ],
                        }
                    )

                    if st.session_state.y_true is not None:
                        results_df["True Label"] = st.session_state.y_true

                    # Show Metrics if Ground Truth exists
                    if st.session_state.y_true is not None:
                        acc = skm.accuracy_score(st.session_state.y_true, y_pred)
                        prec = skm.precision_score(
                            st.session_state.y_true, y_pred, zero_division=0
                        )
                        rec = skm.recall_score(
                            st.session_state.y_true, y_pred, zero_division=0
                        )
                        f1 = skm.f1_score(
                            st.session_state.y_true, y_pred, zero_division=0
                        )

                        st.markdown("#### Performance Metrics")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{acc:.3f}")
                        m2.metric("Precision", f"{prec:.3f}")
                        m3.metric("Recall", f"{rec:.3f}")
                        m4.metric("F1 Score", f"{f1:.3f}")

                        st.markdown("#### Detailed Analysis")
                        t_tab1, t_tab2 = st.tabs(["Visualization", "Prediction Data"])

                        with t_tab1:
                            threshold = st.slider(
                                "Probability Threshold", 0.0, 1.0, 0.5, 0.01
                            )
                            model = load_model(model_selection)
                            evaluator = ModelEvaluator(
                                model=model,
                                X=st.session_state.X_scaled,
                                Y=st.session_state.y_true,
                                threshold=threshold,
                                model_name=model_selection,
                                figsize=(12, 10),
                            )
                            evaluator.plot_evaluation()

                        with t_tab2:
                            st.dataframe(results_df, width="stretch")

                    else:
                        # No Ground Truth - Just Predictions
                        st.markdown("#### Prediction Results")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Rows", len(results_df))
                        c2.metric("Predicted Failure", int(sum(y_pred == 1)))
                        c3.metric("Predicted Normal", int(sum(y_pred == 0)))
                        st.dataframe(results_df, width="stretch")

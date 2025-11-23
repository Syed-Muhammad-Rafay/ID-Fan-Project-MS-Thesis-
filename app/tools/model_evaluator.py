###################################################################################################################################
# A class for evaluating and visualizing machine learning model performance using various metrics.
#
#   It provides methods to calculate and plot confusion matrix, ROC curve, precision-recall curve, and classification report
###################################################################################################################################

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)


class ModelEvaluator:
    def __init__(
        self,
        model,
        X,
        Y,
        threshold=0.5,
        model_name=None,
        figsize=(14, 12),
        history=None,
        history_figsize=(14, 5),
    ) -> None:
        self.model = model
        self.X = X
        self.Y = Y
        self.model_name = model_name
        self.history = history
        self.threshold = threshold
        self.figsize = figsize
        self.history_figsize = history_figsize
        self.primaryColor = "#4fd8a0"  # Light green (primary color)
        self.secondaryColor = "#1f3a60"

        self.y_pred_proba = self.model.predict_proba(self.X)[:, 1]

        self.y_pred = (self.y_pred_proba >= self.threshold).astype(int)

    def _conf_matrix(self):
        cm = confusion_matrix(self.Y, self.y_pred)
        true_positives = np.diag(cm)
        total_actual = np.sum(cm, axis=1)
        cm_percent = np.zeros_like(cm, dtype=float)
        for i in range(len(cm)):
            if total_actual[i] > 0:
                cm_percent[i, i] = true_positives[i] / total_actual[i]

        return cm, cm_percent

    def _roc_curve_auc(self):
        fpr_1, tpr_1, _ = roc_curve(self.Y, self.y_pred_proba)
        roc_auc_1 = auc(fpr_1, tpr_1)

        fpr_0, tpr_0, _ = roc_curve(1 - self.Y, 1 - self.y_pred_proba)
        roc_auc_0 = auc(fpr_0, tpr_0)

        return {
            "class_1": (fpr_1, tpr_1, roc_auc_1),
            "class_0": (fpr_0, tpr_0, roc_auc_0),
        }

    def _pr_curve_auc(self):
        precision_1, recall_1, _ = precision_recall_curve(self.Y, self.y_pred_proba)
        pr_auc_1 = average_precision_score(self.Y, self.y_pred_proba)

        precision_0, recall_0, _ = precision_recall_curve(
            1 - self.Y, 1 - self.y_pred_proba
        )
        pr_auc_0 = average_precision_score(1 - self.Y, 1 - self.y_pred_proba)

        return {
            "class_1": (precision_1, recall_1, pr_auc_1),
            "class_0": (precision_0, recall_0, pr_auc_0),
        }

    def _class_report_df(self):
        class_report = classification_report(self.Y, self.y_pred, output_dict=True)
        report_data = []
        label_map = {"0": "Normal", "1": "Towards Failure"}
        for label, metrics in class_report.items():
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                display_label = label_map.get(label, label)
                row = [display_label]
                row += [
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1-score"],
                    metrics["support"],
                ]
                report_data.append(row)

        df = pd.DataFrame(
            report_data, columns=["Class", "Precision", "Recall", "F1-score", "Support"]
        )
        df.set_index("Class", inplace=True)

        return df

    def plot_evaluation(self) -> None:
        cm, cm_percent = self._conf_matrix()
        roc_data = self._roc_curve_auc()
        pr_data = self._pr_curve_auc()

        fpr_1, tpr_1, roc_auc_1 = roc_data["class_1"]
        fpr_0, tpr_0, roc_auc_0 = roc_data["class_0"]

        precision_1, recall_1, pr_auc_1 = pr_data["class_1"]
        precision_0, recall_0, pr_auc_0 = pr_data["class_0"]

        df = self._class_report_df()

        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        fontsize = self.figsize[1] / 1.5

        if self.model_name is not None:
            title = f"Results for {self.model_name}"
        else:
            title = "Evaluation Results"

        fig.suptitle(title, fontsize=fontsize * 1.5, fontweight="bold")

        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            cm,
            annot=True,
            cmap="Greens",
            fmt="d",
            linecolor="black",
            linewidths=0.7,
            ax=ax0,
            cbar=False,
            xticklabels=["Normal", "Towards Failure"],
            yticklabels=["Normal", "Towards Failure"],
        )
        ax0.set_xlabel("Predicted", fontsize=fontsize)
        ax0.set_ylabel("Actual", fontsize=fontsize)
        ax0.set_title("Confusion Matrix", fontsize=fontsize, fontweight="bold")

        ax1 = fig.add_subplot(gs[0, 1])
        sns.heatmap(
            df.iloc[:, :-1],
            annot=True,
            cmap="Greens",
            fmt=".2f",
            cbar=False,
            linewidths=0.7,
            linecolor="black",
            annot_kws={"weight": "bold", "fontsize": fontsize},
            ax=ax1,
        )
        ax1.set_title("Classification Report", fontsize=fontsize, fontweight="bold")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(
            fpr_1,
            tpr_1,
            lw=2,
            color=self.primaryColor,
            label=f"Towards Failure ROC (AUC = {roc_auc_1:.2f})",
        )
        ax2.plot(
            fpr_0,
            tpr_0,
            lw=2,
            color=self.secondaryColor,
            label=f"Normal ROC (AUC = {roc_auc_0:.2f})",
        )
        ax2.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        ax2.set_xlim((0.0, 1.0))
        ax2.set_ylim((0.0, 1.05))
        ax2.set_xlabel("False Positive Rate", fontsize=fontsize)
        ax2.set_ylabel("True Positive Rate", fontsize=fontsize)
        ax2.set_title("ROC Curves", fontsize=fontsize, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=fontsize)
        ax2.grid(False)

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(
            recall_1,
            precision_1,
            lw=2,
            color=self.primaryColor,
            label=f"Towards Failure PR (AUC = {pr_auc_1:.2f})",
        )
        ax3.plot(
            recall_0,
            precision_0,
            lw=2,
            color=self.secondaryColor,
            label=f"Normal PR (AUC = {pr_auc_0:.2f})",
        )
        ax3.set_xlim((0.0, 1.0))
        ax3.set_ylim((0.0, 1.05))
        ax3.set_xlabel("Recall", fontsize=fontsize)
        ax3.set_ylabel("Precision", fontsize=fontsize)
        ax3.set_title("Precision-Recall Curves", fontsize=fontsize, fontweight="bold")
        ax3.legend(loc="lower left", fontsize=fontsize)
        ax3.grid(False)

        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

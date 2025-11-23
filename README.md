ID Fan Failure Prediction
=========================

### What's changed?

1. added two models for a comparison ( LightGBM, and Extra Trees )
2. added F1 Score metric in the results dataframe
3. created proper evaluation visualizations, for ROC AUC, PR AUC, Confusion Matrix, and Classification Reports for the selected models

All results and output can be found in [output](./output/) directory:

```sh
output/
├── ckpts
│   ├── classification_report.png
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   ├── train_and_test_distribution.png
│   └── *.joblib                            <- all model's saved as joblib
│
├── eda
│   ├── correlation_heatmap.png
│   ├── correlation_with_target_class.png
│   ├── normalized_vibration_and_target_class.png
│   └── power_and_target_class.png
│
├── processed_id_fan_data.csv
└── results.csv

```


The notebook can be run locally or in Colab environment:

### How to run locally?

1. If you have `uv` installed, run:

```sh
uv sync
```
2. If not, create a virtual environment and activate it:

```sh
# create a venv
python3 -m venv venv

# activate: Linux/macOS
source venv/bin/activate

# activate: Windows
venv\Scripts\activate

```

3. Install dependencies:
```sh
pip install -r requirements.txt
```

4. Open the [notebook](./notebooks/main.ipynb), and run all cells.

### How to run Streamlit app locally?

1. Follow steps 1-3 from "How to run locally?" above to set up your environment and install dependencies.

2. Run the Streamlit app:
```sh
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### How to run on Colab?

Go to this [link](https://colab.research.google.com/github/Syed-Muhammad-Rafay/ID-Fan-Project-MS-Thesis-/blob/main/notebooks/main.ipynb), and run all cells
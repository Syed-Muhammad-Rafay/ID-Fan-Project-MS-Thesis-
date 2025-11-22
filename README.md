ID Fan Failure Prediction
=========================

### What's changed?

1. added two models for a comparison ( LightGBM, and Extra Trees )
2. added F1 Score metric in the results dataframe
3. created proper evaluation visualizations, for ROC AUC, PR AUC, Confusion Matrix, and Classification Reports for the selected models

All results and output can be found in [output](./output/) directory and they can be run locally or in Colab environment

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

3. Open the [notebook](./notebooks/main.ipynb), and run all cells.

### How to run on Colab?

Go to this [link](https://colab.research.google.com/github/ahmedsalim3/id-fan-failure-prediction/blob/main/notebooks/main.ipynb), and run all cells
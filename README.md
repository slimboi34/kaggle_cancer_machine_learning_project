# Histopathologic Cancer Detection

This repository contains the solution for the Kaggle competition **Histopathologic Cancer Detection**.
The problem involves identifying metastatic cancer in small image patches (96x96 pixels) taken from larger digital pathology scans.

This solution implements a unique **Center-Focus Dual-Path CNN** to address the specific framing of this competition: a positive label implies tumor tissue exists in the *center 32x32px region* of the image patch.

## Contents
- `histopathologic_cancer_detection.ipynb`: A complete Jupyter Notebook covering Problem Description, Exploratory Data Analysis (EDA), Model Architecture, Results, and Conclusions.
- `requirements.txt`: Required dependencies.

## Instructions
1. Download the dataset from the [competition page](https://www.kaggle.com/c/histopathologic-cancer-detection/data).
2. Place the `train` and `test` directories along with `train_labels.csv` inside an `input/` folder in the respective directory or upload the notebook to Kaggle directly.
3. Install the dependencies using `pip install -r requirements.txt`.
4. Run the Jupyter Notebook cells sequentially.

## Future Work
- Incorporating transfer-learning for the context stream (e.g., EfficientNet).
- Using Test-Time Augmentation (TTA) to improve robust predictions on the test set.
# kaggle_cancer_machine_learning_project

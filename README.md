# Flower Classification

This project focuses on classifying different types of flowers using machine learning models. The models implemented include SVM (Support Vector Machine), LightGBM (Light Gradient Boosting Machine), and Random Forest.

## Project Structure

```bash
├── Data
│   ├── train
│   ├── val
│   └── test
├── Finetunning_result
│   ├── RandomForest.csv
│   ├── lightGBM.csv
│   └── svm.csv
├── pickle
│   ├── svm_model_4.pkl
│   ├── lightgbm_model_4.pkl
│   ├── rf_model_4.pkl
│   ├── scaler_4.pkl
│   ├── pca_4.pkl
│   └── encoder_4.pkl
├── .gitattributes
├── Demo.ipynb
├── Finetunning.ipynb
├── demo-app.py
└── training_model.py
```

## Installation

To get started with this project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

## Data

The dataset is organized into three folders: `train`, `val`, and `test`. Each folder contains subfolders named after the flower types (e.g., `daisy`, `sunflower`). The images should be placed in their respective subfolders.

## Training Models

The `training_model.py` script is used to train the models. It extracts features using HOG (Histogram of Oriented Gradients) and Color Histograms, reduces dimensionality with PCA, and trains SVM, LightGBM, and Random Forest models.

### Usage

Run the script to train the models:

```bash
python training_model.py
```
Alternatively, you can use the pre-trained models available in the pickle folder.

### Training Process

1. **Extract Features**:
   - HOG features
   - Color Histogram

2. **Dimensionality Reduction**:
   - PCA

3. **Train Models**:
   - SVM
   - LightGBM
   - Random Forest

4. **Evaluate Models**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

## Fine-Tuning

The `Finetunning.ipynb` notebook contains the process for fine-tuning the hyperparameters of the models. It evaluates different configurations and selects the best performing parameters.

## Demo Application

The `demo-app.py` script is a Streamlit application that allows users to upload images of flowers and get predictions from the trained models.

### Usage

Run the Streamlit application:

```bash
streamlit run demo-app.py
```

### Features

- Upload multiple images of flowers
- Get predictions from SVM, LightGBM, and Random Forest models
- View HOG and Color Histogram visualizations

## Results

The `Finetunning_result` folder contains the final results of the model fine-tuning process.

### Performance Metrics

The performance of the models is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score

### Best Model

The SVM model with an RBF kernel and `C=15` performed the best with the highest accuracy on the validation and test sets.

## Contributors

- Châu Thế Vĩ (22521653)
- Lê Bình Nguyên (22520969)

## Acknowledgments

This project was guided by TS. Mai Tiến Dũng as part of the CS231 course at the University of Information Technology, Vietnam National University, Ho Chi Minh City.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

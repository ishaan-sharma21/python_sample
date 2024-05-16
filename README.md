## Description
This project aims to predict the diagnosis (malignant or benign) of breast masses using features characterizing cell nuclei. The dataset used for training and evaluation contains measurements of various features extracted from digitized images of breast mass fine needle aspirate (FNA) samples.

## Dataset
The dataset (`data.csv`) consists of the following:
- **id**: Unique identifier for each sample
- **diagnosis**: The diagnosis label indicating whether the sample is malignant (M) or benign (B)
- **Features**: Numerical values representing various characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension, measured for mean, standard error, and "worst" or largest mean value.

## Script Functionality
The Python script (`analysis.py`) performs the following tasks:
1. **Data Loading**: Reads the dataset (`data.csv`) containing breast cancer diagnostic features.
2. **Data Visualization**: Generates heatmaps to visualize correlations between features for malignant and benign diagnoses.
3. **Model Training and Evaluation**: Performs logistic regression to predict breast cancer diagnosis based on selected features. It splits the dataset into training and testing sets, fits a logistic regression model, makes predictions, prints coefficients, and evaluates the model's accuracy using metrics like accuracy score and confusion matrix.

## Installation
1. Clone or download the project repository.
2. Ensure Python and the required libraries (Pandas, Matplotlib, Seaborn, and scikit-learn) are installed.

## Usage
1. Place your dataset file (`data.csv`) in the same directory as the script.
2. Run the script using Python:
   ```
   python analysis.py
   ```

## Credits
This data set was obtained via Kaggle, uploaded by UCI Machine Learning.

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

## Contact
For any questions or inquiries, please contact Ishaan Sharma at eesharma21@gmail.com.

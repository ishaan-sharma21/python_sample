import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"C:\Users\eesha\Desktop\morg-proj\data.csv")

malignant_df = df[df['diagnosis'] == 'M']
benign_df = df[df['diagnosis'] == 'B']

def read_and_print_first_5():
    print(df.head())

def visualize_most_important_vars(df):
    features = list(df.columns[2:])

    plt.figure(figsize=(15, 10))
    sns.heatmap(malignant_df[features].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap for Malignant Diagnosis')
    plt.savefig("malig.pdf")

    plt.figure(figsize=(15, 10))
    sns.heatmap(benign_df[features].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap for Benign Diagnosis')
    plt.savefig("benig.pdf")

def perform_regression():
     # Selecting the mean-features
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
    
    X = df[features]
    y = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convert diagnosis to binary (0: benign, 1: malignant)
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initializing the logistic regression model
    model = LogisticRegression(max_iter=1000)
    
    # Fitting the model on the training data
    model.fit(X_train, y_train)
    
    # Making predictions on the test data
    y_pred = model.predict(X_test)
    coef = model.coef_[0]
    feature_names = X.columns
    
    # Printing coefficients and feature names
    print("Feature Coefficients:")
    for feature, coef in zip(feature_names, coef):
        print(feature, ':', coef)
    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_mat)

def main():
    read_and_print_first_5()
    visualize_most_important_vars(df)
    perform_regression()

if __name__ == "__main__":  
    main()
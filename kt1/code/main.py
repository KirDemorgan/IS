import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv('e:/Kt_1_int_sys/kt2/1/telecom_churn.csv')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Exploration
    print("\n--- Data Exploration ---")
    print("Dataset shape:", df.shape)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    print(f"Numerical columns ({len(numerical_cols)}):", numerical_cols)
    print(f"Categorical columns ({len(categorical_cols)}):", categorical_cols)
    print("\nMissing values:\n", df.isnull().sum())

    # Target variable distribution
    print("\nChurn distribution:\n", df['Churn'].value_counts())

    # Data Preparation
    print("\n--- Data Preparation ---")
    # Separate features and target
    X = df.drop('Churn', axis=1)
    # Map target variable to int if it's boolean
    y = df['Churn'].astype(int)

    # Define numerical and categorical features for the models
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }

    results = {}

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Create pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        # Train
        clf.fit(X_train, y_train)
        # Predict
        y_pred = clf.predict(X_test)
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")

    # Plotting results
    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color=['blue', 'green', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('e:/Kt_1_int_sys/kt1/code/accuracy_comparison.png')
    print("\nAccuracy comparison plot saved as 'accuracy_comparison.png'")

    # Best model analysis
    best_model_name = max(results, key=results.get)
    print(f"\n--- Best Model Analysis ---")
    print(f"Best Model based on Accuracy: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")

    best_model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', models[best_model_name])])
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)
    print("\nConfusion Matrix for Best Model:\n", cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loyal (0)', 'Churned (1)'], yticklabels=['Loyal (0)', 'Churned (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {best_model_name}')
    plt.tight_layout()
    plt.savefig('e:/Kt_1_int_sys/kt1/code/confusion_matrix.png')
    print("Confusion matrix plot saved as 'confusion_matrix.png'")

    # Interpretation
    # Assuming False (0) is Loyal and True (1) is Churned
    true_negatives = cm[0, 0]  # Predicted Loyal, Actual Loyal
    false_positives = cm[0, 1] # Predicted Churned, Actual Loyal
    false_negatives = cm[1, 0] # Predicted Loyal, Actual Churned
    true_positives = cm[1, 1]  # Predicted Churned, Actual Churned

    print(f"\nInterpretation:")
    print(f"Model correctly predicted {true_positives} as churned and {true_negatives} as loyal.")
    print(f"Model erroneously predicted as churned (False Positives): {false_positives} (Clients predicted as churned, but actually loyal)")
    print(f"Model erroneously predicted as loyal (False Negatives): {false_negatives} (Clients predicted as loyal, but actually churned)")


if __name__ == "__main__":
    main()

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
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("Загрузка датасета...")
    try:
        df = pd.read_csv('telecom_churn.csv')
    except Exception as e:
        print(f"Ошибка загрузки CSV: {e}")
        return

    print("\n--- Разбор данных ---")
    print("Размер датасета (строки, столбцы):", df.shape)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    print(f"Числовые столбцы ({len(numerical_cols)}):", numerical_cols)
    print(f"Категориальные столбцы ({len(categorical_cols)}):", categorical_cols)
    print("\nПропущенные значения:\n", df.isnull().sum())

    print("\nРаспределение целевой переменной (ушел ли клиент):\n", df['churn'].value_counts())

    print("\n--- Подготовка данных к обучению ---")
    X = df.drop('churn', axis=1)
    y = df['churn'].astype(int)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Размер обучающей выборки: {X_train.shape[0]} строк")
    print(f"Размер тестовой выборки: {X_test.shape[0]} строк")

    models = {
        'Логистическая регрессия': LogisticRegression(random_state=42, max_iter=1000),
        'Дерево решений': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }

    results = {}

    print("\n--- Обучение и проверка модели ---")
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"Точность (Accuracy) модели '{name}': {acc:.4f}")

    names = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color=['blue', 'green', 'orange'])
    plt.ylabel('Точность (Accuracy)')
    plt.title('Сравнение точности моделей')
    plt.ylim(0, 1.0)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    print("\nГрафик сравнения точности сохранен как 'accuracy_comparison.png'")

    best_model_name = max(results, key=results.get)
    print(f"\n--- Анализ результатов ---")
    print(f"Лучшая модель на данном этапе: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")

    best_model = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', models[best_model_name])])
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)
    print("\nМатрица ошибок (confusion matrix) для лучшей модели:\n", cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Лояльные (0)', 'Ушедшие (1)'], yticklabels=['Лояльные (0)', 'Ушедшие (1)'])
    plt.xlabel('Предсказанные')
    plt.ylabel('Фактические')
    plt.title(f'Матрица ошибок: {best_model_name}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("График матрицы ошибок сохранен как 'confusion_matrix.png'")

    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    print(f"\nИнтерпретация:")
    print(f"Сколько клиентов модель ошибочно предсказала как ушедших: {false_positives}")
    print(f"Сколько клиентов модель ошибочно предсказала как лояльных: {false_negatives}")

if __name__ == "__main__":
    main()

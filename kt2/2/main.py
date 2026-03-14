import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=== Практическое занятие № 2: Настройка и улучшение ML-модели ===\n")
    
    print("Загрузка датасета...")
    try:
        df = pd.read_csv('telecom_churn.csv')
    except Exception as e:
        print(f"Ошибка загрузки CSV: {e}")
        return

    X = df.drop('churn', axis=1)
    y = df['churn'].astype(int)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ])

    print("\n--- 1. Диагностика проблемы старой модели ---")
    old_model = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    old_model.fit(X_train, y_train)
    y_pred_old = old_model.predict(X_test)
    
    print("\nОтчет о классификации (Старая модель):")
    print(classification_report(y_test, y_pred_old, target_names=['Лояльные (0)', 'Ушедшие (1)']))
    
    old_acc = accuracy_score(y_test, y_pred_old)
    old_prec = precision_score(y_test, y_pred_old)
    old_rec = recall_score(y_test, y_pred_old)
    old_f1 = f1_score(y_test, y_pred_old)
    
    cm_old = confusion_matrix(y_test, y_pred_old)
    disp_old = ConfusionMatrixDisplay(confusion_matrix=cm_old, display_labels=['Лояльные (0)', 'Ушедшие (1)'])
    disp_old.plot(cmap='Blues', values_format='d')
    plt.title("Матрица ошибок: Старая модель (Decision Tree)")
    plt.tight_layout()
    plt.savefig('confusion_matrix_old.png')
    plt.close()
    
    print("Матрица ошибок старой модели сохранена: 'confusion_matrix_old.png'")
    
    print("\nАнализ классов:\nВ выборке присутствует сильный дисбаланс: лояльных клиентов подавляющее большинство. " 
          "Ошибки False Negative (когда мы не предсказали уход клиента, а он ушел) обходятся бизнесу намного дороже, "
          "так как мы теряем доход. Наша цель – повысить показатель Recall для ушедших клиентов (класс 1).")

    print("\n--- 2. Системный поиск лучших параметров (GridSearchCV + SMOTE) ---")
    
    new_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__max_depth': [5, 10, 15, 20, None],
        'classifier__min_samples_split': [2, 10, 20],
        'classifier__min_samples_leaf': [1, 5, 10],
        'classifier__criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        new_pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1', 
        n_jobs=-1,
        verbose=1
    )
    
    print("Начат поиск оптимальных параметров по сетке с 5-Fold кросс-валидацией...")
    grid_search.fit(X_train, y_train)
    
    print("\nЛучшая комбинация гиперпараметров (best_params_):")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
    
    print(f"Лучший результат кросс-валидации (F1 best_score_): {grid_search.best_score_:.4f}")

    print("\n--- 3. Финальная оценка улучшенной модели ---")
    
    best_model = grid_search.best_estimator_
    
    y_pred_new = best_model.predict(X_test)
    y_prob_new = best_model.predict_proba(X_test)[:, 1]
    
    new_acc = accuracy_score(y_test, y_pred_new)
    new_prec = precision_score(y_test, y_pred_new)
    new_rec = recall_score(y_test, y_pred_new)
    new_f1 = f1_score(y_test, y_pred_new)
    
    print("Отчет о классификации (Новая модель):")
    print(classification_report(y_test, y_pred_new, target_names=['Лояльные (0)', 'Ушедшие (1)']))

    cm_new = confusion_matrix(y_test, y_pred_new)
    disp_new = ConfusionMatrixDisplay(confusion_matrix=cm_new, display_labels=['Лояльные (0)', 'Ушедшие (1)'])
    disp_new.plot(cmap='Blues', values_format='d')
    plt.title("Матрица ошибок: Оптимизированная модель")
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized.png')
    plt.close()
    print("Матрица ошибок новой модели сохранена: 'confusion_matrix_optimized.png'")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob_new)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Кривая (Receiver Operating Characteristic)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()
    print(f"ROC-кривая сохранена: 'roc_curve.png' (AUC-ROC: {roc_auc:.3f})")

    print("\n--- Важность признаков (Feature Importance) ---")
    fitted_preprocessor = best_model.named_steps['preprocessor']
    
    num_feat_names = numeric_features.tolist()
    
    ohe = fitted_preprocessor.named_transformers_['cat']
    cat_feat_names = ohe.get_feature_names_out(categorical_features).tolist()
    
    all_feature_names = num_feat_names + cat_feat_names
    
    classifier = best_model.named_steps['classifier']
    importances = classifier.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    
    print("Топ-10 самых важных признаков:")
    for i in range(min(10, len(indices))):
        print(f"{i+1}. {all_feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    
    top_n = min(15, len(indices))
    plt.figure(figsize=(10, 8))
    plt.title("Топ-15 самых важных признаков")
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], color='steelblue', align='center')
    plt.yticks(range(top_n), [all_feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel("Относительная важность")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("График важности признаков сохранен: 'feature_importance.png'")

    print("\n=======================================================")
    print("ИТОГОВОЕ СРАВНЕНИЕ МЕТРИК")
    print("=======================================================")
    
    import warnings
    warnings.filterwarnings('ignore')
    
    data = [
        ['Accuracy (Точность)', f"{old_acc:.4f}", f"{new_acc:.4f}"],
        ['Precision', f"{old_prec:.4f}", f"{new_prec:.4f}"],
        ['Recall (Ушедшие)', f"{old_rec:.4f}", f"{new_rec:.4f}"],
        ['F1-Score', f"{old_f1:.4f}", f"{new_f1:.4f}"],
        ['ROC-AUC', '-', f"{roc_auc:.4f}"]
    ]
    comp_df = pd.DataFrame(data, columns=['Метрика', 'Старая модель', 'Новая (Оптимизированная)'])
    print(comp_df.to_string(index=False))
    print("\nЦель: Повысить способность модели находить уходящих клиентов (Recall) при сохранении приемлемого уровня общего качества.")

if __name__ == "__main__":
    main()

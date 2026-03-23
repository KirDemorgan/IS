"""
train.py — Скрипт обучения модели классификации видов ирисов.

Блок 1: Загрузка и подготовка данных
Блок 2: Обучение двух моделей + кросс-валидация + анализ ошибок
Блок 3: Выбор лучшей модели и сохранение в файл
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

warnings.filterwarnings('ignore')

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# Модель 1: собственная реализация — пороговый классификатор по одному признаку
class SimpleThresholdClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, feature_index=2):
        self.feature_index = feature_index  # индекс petal length

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            petal_length = X.iloc[:, self.feature_index].values
        else:
            petal_length = X[:, self.feature_index]

        y = np.array(y)
        self.classes_ = np.unique(y)

        best_accuracy = 0
        best_t1, best_t2 = 2.5, 4.8

        for t1 in np.arange(1.0, 4.0, 0.1):
            for t2 in np.arange(t1 + 0.5, 7.0, 0.1):
                preds = np.where(petal_length < t1, 0,
                         np.where(petal_length < t2, 1, 2))
                acc = np.mean(preds == y)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_t1, best_t2 = t1, t2

        self.threshold_1_ = best_t1
        self.threshold_2_ = best_t2
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            petal_length = X.iloc[:, self.feature_index].values
        else:
            petal_length = X[:, self.feature_index]

        return np.where(petal_length < self.threshold_1_, 0,
                np.where(petal_length < self.threshold_2_, 1, 2))


def main():
    print("=" * 60)
    print("  ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ВИДОВ ИРИСОВ")
    print("=" * 60)

    # === БЛОК 1: ПОДГОТОВКА ДАННЫХ ===
    print("\n--- Блок 1: Подготовка данных ---\n")

    data_path = os.path.join(PROJECT_DIR, 'data', 'iris.csv')
    df = pd.read_csv(data_path)

    print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # X — 4 числовых признака (измерения цветка)
    # y — вид ириса (0=setosa, 1=versicolor, 2=virginica)
    X = df.drop('species', axis=1)
    y = df['species']

    print(f"Признаки (X): {list(X.columns)}")
    print(f"Целевая переменная (y): species")
    print(f"Все признаки числовые — кодирование категориальных не требуется")

    scaler = StandardScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Разделение: {len(X_train)} обучение / {len(X_test)} тест (80/20, stratify)")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Блок 2: Обучение и диагностика ---\n")

    # Модель 1: собственная реализация (пороговый классификатор по petal length)
    print("Модель 1: Пороговый классификатор (собственная реализация)")
    model_simple = SimpleThresholdClassifier()
    model_simple.fit(X_train, y_train)
    print(f"  Пороги: petal_length < {model_simple.threshold_1_:.1f} → setosa | "
          f"< {model_simple.threshold_2_:.1f} → versicolor | иначе → virginica")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_simple = cross_val_score(model_simple, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  Кросс-валидация (5-Fold): Accuracy = {scores_simple.mean():.4f} ± {scores_simple.std():.4f}")

    y_pred_simple = model_simple.predict(X_test)
    print(f"\n  Classification Report (тест):")
    print(classification_report(y_test, y_pred_simple, target_names=CLASS_NAMES, digits=4))

    # Модель 2: RandomForestClassifier (sklearn) — ансамбль из 100 деревьев
    print("Модель 2: RandomForestClassifier (sklearn)")
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    scores_rf = cross_val_score(model_rf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    print(f"  Кросс-валидация (5-Fold): Accuracy = {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")

    y_pred_rf = model_rf.predict(X_test_scaled)
    print(f"\n  Classification Report (тест):")
    print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, digits=4))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_pred, title in [
        (axes[0], y_pred_simple, 'Пороговый классификатор'),
        (axes[1], y_pred_rf, 'RandomForest')
    ]:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_xlabel('Предсказание')
        ax.set_ylabel('Реальный класс')
        ax.set_title(f'Confusion Matrix — {title}')
    plt.tight_layout()
    fig.savefig(os.path.join(PROJECT_DIR, 'confusion_matrices.png'), dpi=150)
    plt.close()
    print("  Графики сохранены: confusion_matrices.png")

    print("\n--- Анализ ошибок ---")
    for name, y_pred in [('Пороговый', y_pred_simple), ('RandomForest', y_pred_rf)]:
        errors = y_test.values != y_pred
        if errors.sum() > 0:
            cm = confusion_matrix(y_test, y_pred)
            np.fill_diagonal(cm, 0)
            max_idx = np.unravel_index(cm.argmax(), cm.shape)
            print(f"  {name}: {errors.sum()} ошибок. "
                  f"Чаще всего путает {CLASS_NAMES[max_idx[0]]} → {CLASS_NAMES[max_idx[1]]}")
        else:
            print(f"  {name}: 0 ошибок на тестовой выборке!")

    print("\n--- Блок 3: Финальный отбор и сохранение ---\n")

    acc_simple = accuracy_score(y_test, y_pred_simple)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_simple = f1_score(y_test, y_pred_simple, average='macro')
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')

    print(f"  {'Модель':<30} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Пороговый (собственный)':<30} {acc_simple:>10.4f} {f1_simple:>10.4f}")
    print(f"  {'RandomForest (sklearn)':<30} {acc_rf:>10.4f} {f1_rf:>10.4f}")

    if f1_rf >= f1_simple:
        best_model, best_name, best_acc, best_f1, best_pred = model_rf, 'RandomForest', acc_rf, f1_rf, y_pred_rf
    else:
        best_model, best_name, best_acc, best_f1, best_pred = model_simple, 'Пороговый классификатор', acc_simple, f1_simple, y_pred_simple

    models_dir = os.path.join(PROJECT_DIR, 'models')
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"\n  Модель сохранена: models/best_model.pkl")
    print(f"  Scaler сохранён: models/scaler.pkl")

    cm_best = confusion_matrix(y_test, best_pred)
    np.fill_diagonal(cm_best, 0)
    if cm_best.sum() > 0:
        max_idx = np.unravel_index(cm_best.argmax(), cm_best.shape)
        confusion_msg = f"Чаще всего путает {CLASS_NAMES[max_idx[0]]} и {CLASS_NAMES[max_idx[1]]}"
    else:
        confusion_msg = "Ошибок на тестовых данных нет"

    print("\n" + "=" * 60)
    print(f"  ИТОГ: Лучшая модель — {best_name}.")
    print(f"  Accuracy = {best_acc:.4f}, Macro-F1 = {best_f1:.4f}.")
    print(f"  {confusion_msg}.")
    print("=" * 60)


if __name__ == '__main__':
    main()

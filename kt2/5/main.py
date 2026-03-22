import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    print("=== Практическое занятие № 5: Прототип оценки стоимости домов ===\n")

    print("Загрузка датасета California Housing...")
    try:
        california = fetch_california_housing()
        X_full = pd.DataFrame(california.data, columns=california.feature_names)
        y = pd.Series(california.target, name='MedHouseVal')
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        return

    print("\n--- 1. Исследование данных и целевой переменной ---")
    plt.figure(figsize=(8, 6))
    sns.histplot(y, bins=50, kde=True, color='teal')
    plt.title('Распределение медианной стоимости домов (MedHouseVal)')
    plt.xlabel('Стоимость (в 100 000 $)')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('target_histogram.png')
    plt.close()
    print("Гистограмма целевой переменной сохранена: 'target_histogram.png'")

    selected_features = ['MedInc', 'AveRooms']
    print(f"\nВыбраны признаки для анализа: {selected_features}")
    print(" - MedInc: Медианный доход в районе")
    print(" - AveRooms: Среднее количество комнат")

    X = X_full[selected_features]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, feature in enumerate(selected_features):
        sns.scatterplot(x=X[feature], y=y, alpha=0.2, ax=axes[i], color='indigo')
        axes[i].set_title(f'Связь: {feature} vs Стоимость')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Стоимость (в 100k $)')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('features_vs_target.png')
    plt.close()
    print("Графики рассеяния сохранены: 'features_vs_target.png'")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("\n--- 2. Обучение и оценка линейной модели ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f"Линейная регрессия (LinearRegression):")
    print(f"  MAE: {mae_lr:.4f}")
    print(f"  MSE: {mse_lr:.4f}")
    print(f"  R2:  {r2_lr:.4f}")

    print("\nИнтерпретация коэффициентов:")
    for feature, coef in zip(selected_features, lr.coef_):
        print(f"  Коэффициент [{feature}]: {coef:.4f}")
    print(f"  Intercept (Смещение): {lr.intercept_:.4f}")
    print("\nБизнес-интерпретация для заказчика:")
    print(f"Увеличение показателя 'AveRooms' (средняя комнатность) на 1 единицу приводит "
          f"к изменению прогнозируемой стоимости дома в среднем на {lr.coef_[1] * 100:.2f} тысяч долларов "
          f"(при условии неизменности дохода).")

    print("\n--- 3. Сравнение с нелинейной моделью ---")
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)

    print(f"Дерево решений (DecisionTreeRegressor, max_depth=3):")
    print(f"  MAE: {mae_dt:.4f}")
    print(f"  R2:  {r2_dt:.4f}")

    print("\n=======================================================")
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=======================================================")
    data = [
        ['LinearRegression', f"{mae_lr:.4f}", f"{r2_lr:.4f}"],
        ['DecisionTree (depth=3)', f"{mae_dt:.4f}", f"{r2_dt:.4f}"]
    ]
    comp_df = pd.DataFrame(data, columns=['Модель', 'MAE', 'R2 Score'])
    print(comp_df.to_string(index=False))

    best_model_name = "LinearRegression" if r2_lr > r2_dt else "DecisionTreeRegressor"
    best_preds = y_pred_lr if best_model_name == "LinearRegression" else y_pred_dt
    print(f"\nДля финального графика выбрана модель: {best_model_name}")

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, best_preds, alpha=0.2, color='darkred')
    # Идеальная прямая y = x
    min_val = min(y.min(), best_preds.min())
    max_val = max(y.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'b--', lw=2, label='Идеальное предсказание (y=x)')
    plt.title(f'Реальная цена vs Предсказанная ({best_model_name})')
    plt.xlabel('Реальная стоимость (в 100k $)')
    plt.ylabel('Предсказанная стоимость (в 100k $)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('real_vs_pred_scatter.png')
    plt.close()
    print("График сравнения предсказаний (Scatter plot) сохранен: 'real_vs_pred_scatter.png'")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import sys
import io

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main():
    print("=== Практическое занятие № 6: Борьба с переобучением (Ridge) ===\n")

    print("--- 1. Подготовка базовых данных ---")
    california = fetch_california_housing()
    X = california.data
    y = california.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_base = LinearRegression()
    lr_base.fit(X_train_scaled, y_train)
    r2_base_test = r2_score(y_test, lr_base.predict(X_test_scaled))
    print(f"Базовый R2 на тестовой выборке (ориентир): {r2_base_test:.4f}")

    print("\n--- 2. Создаем проблему (Добавление шума) ---")
    # Добавляем 30 случайных признаков-шумов
    np.random.seed(42)
    noise_train = np.random.normal(0, 1, (X_train.shape[0], 30))
    noise_test = np.random.normal(0, 1, (X_test.shape[0], 30))

    X_train_noisy = np.hstack((X_train, noise_train))
    X_test_noisy = np.hstack((X_test, noise_test))

    print(f"Размерность данных до шума: {X_train.shape[1]} признаков")
    print(f"Размерность данных после добавления шума: {X_train_noisy.shape[1]} признаков")

    scaler_noisy = StandardScaler()
    X_train_noisy_scaled = scaler_noisy.fit_transform(X_train_noisy)
    X_test_noisy_scaled = scaler_noisy.transform(X_test_noisy)

    print("\n--- 3. Демонстрация поломки (Переобучение) ---")
    lr_broken = LinearRegression()
    lr_broken.fit(X_train_noisy_scaled, y_train)

    r2_broken_train = r2_score(y_train, lr_broken.predict(X_train_noisy_scaled))
    r2_broken_test = r2_score(y_test, lr_broken.predict(X_test_noisy_scaled))

    print(f"R2 на обучающей выборке (с шумом): {r2_broken_train:.4f}")
    print(f"R2 на тестовой выборке (с шумом):  {r2_broken_test:.4f}")
    print(f"ВНИМАНИЕ: Падение качества на тесте составило {r2_base_test - r2_broken_test:.4f} пунктов!")

    print("\n--- 4. Поиск и спасение (Ridge-регрессия) ---")
    alphas = [0.1, 1, 10, 100, 1000]
    best_alpha = None
    best_r2_ridge = -float('inf')

    print("Перебор коэффициентов регуляризации (alpha):")
    for a in alphas:
        ridge = Ridge(alpha=a, random_state=42)
        ridge.fit(X_train_noisy_scaled, y_train)
        r2_test_ridge = r2_score(y_test, ridge.predict(X_test_noisy_scaled))
        print(f"  Alpha = {a:<6} | R2_test = {r2_test_ridge:.4f}")

        if r2_test_ridge > best_r2_ridge:
            best_r2_ridge = r2_test_ridge
            best_alpha = a

    print(f"\nОптимальное значение найдено: alpha = {best_alpha}")

    print("\n=======================================================")
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ И ВЫВОДЫ")
    print("=======================================================")

    data = [
        ['Базовая (предыдущая практика)', f"{r2_base_test:.4f}", 'Наш ориентир'],
        ['Сломанная (LinearReg + шум)', f"{r2_broken_test:.4f}", 'Катастрофа!'],
        [f'Исправленная (Ridge, alpha={best_alpha})', f"{best_r2_ridge:.4f}", 'Спасение!']
    ]
    results_df = pd.DataFrame(data, columns=['Модель/условия', 'R2 на test', 'Вывод'])
    print(results_df.to_string(index=False))

    print("\nОтветы на вопросы:")
    print(f"1. Насколько упало качество после добавления шума? На {r2_base_test - r2_broken_test:.4f} пунктов.")
    print(f"2. При каком alpha Ridge показал лучший результат? При alpha = {best_alpha}.")

    recovered_diff = r2_base_test - best_r2_ridge
    if recovered_diff < 0.01:
        print(
            f"3. Удалось ли вернуть качество к базовому уровню? Да, практически полностью. Разница с базовым уровнем всего {recovered_diff:.4f}.")
    else:
        print(
            f"3. Удалось ли вернуть качество к базовому уровню? Частично. Качество сильно возросло по сравнению со 'сломанной' моделью, но до идеала не хватило {recovered_diff:.4f}.")


if __name__ == "__main__":
    main()
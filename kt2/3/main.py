import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score
)

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("=== Практическое занятие № 3: Прогнозирование нескольких классов ===")
    
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names

    print(f"\nКоличество классов: {len(target_names)} ({', '.join(target_names)})")
    print(f"Количество признаков: {len(feature_names)}")

    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    print("\n--- Базовый EDA ---")
    class_dist = df['target'].value_counts().sort_index()
    print("Распределение классов:")
    for cls, count in class_dist.items():
        print(f" Класс {cls} ({target_names[cls]}): {count} объектов")

    plt.figure(figsize=(8, 5))
    sns.countplot(x='target', data=df, palette='viridis')
    plt.title('Распределение классов в датасете Wine')
    plt.xticks(ticks=[0, 1, 2], labels=target_names)
    plt.xlabel('Класс вина')
    plt.ylabel('Количество')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    print("\nГрафик распределения сохранен как 'class_distribution.png'")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\n--- Обучение и базовая оценка ---")
    
    models = {
        'RandomForest (Base)': RandomForestClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=10000)
    }

    results = {}

    for name, model in models.items():
        print(f"\nМодель: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': acc,
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'report': classification_report(y_test, y_pred, target_names=target_names)
        }
        
        print(f"Общая Accuracy: {acc:.4f}")
        print("Отчет классификации:")
        print(results[name]['report'])
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Матрица ошибок: {name}')
        plt.tight_layout()
        filename = f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Матрица ошибок сохранена как '{filename}'")

    print("\n--- Сравнение и выбор стратегии (Балансировка) ---")
    
    rf_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_balanced.fit(X_train, y_train)
    y_pred_bal = rf_balanced.predict(X_test)
    
    acc_bal = accuracy_score(y_test, y_pred_bal)
    macro_f1_bal = f1_score(y_test, y_pred_bal, average='macro')
    report_bal = classification_report(y_test, y_pred_bal, target_names=target_names)
    
    results['RandomForest (Balanced)'] = {
        'accuracy': acc_bal,
        'macro_f1': macro_f1_bal,
        'report': report_bal
    }

    print(f"\nМодель: RandomForest (С балансировкой весов)")
    print(f"Общая Accuracy: {acc_bal:.4f}")
    print("Отчет классификации:")
    print(report_bal)

    cm_bal = confusion_matrix(y_test, y_pred_bal)
    disp_bal = ConfusionMatrixDisplay(confusion_matrix=cm_bal, display_labels=target_names)
    disp_bal.plot(cmap='Blues', values_format='d')
    plt.title('Матрица ошибок: RandomForest (Balanced)')
    plt.tight_layout()
    plt.savefig('cm_RandomForest_Balanced.png')
    plt.close()

    print("\n=======================================================")
    print("ИТОГОВОЕ СРАВНЕНИЕ (Macro F1-Score)")
    print("=======================================================")
    for name, res in results.items():
        print(f"{name}: {res['macro_f1']:.4f}")

    print("\nАНАЛИЗ И ВЫВОД:")
    print("Анализ матрицы ошибок и отчета:")
    print(" Класс 'class_1' (второй класс) является самым многочисленным (71 объект), а класс 'class_2' (третий класс) — самым редким (48 объектов). Дисбаланс присутствует, но он не экстремальный.")
    print(" У базового RandomForestRecall иногда может проседать на самом маленьком классе (class_2), так как модель оптимизирует общую точность за счет преобладающих классов.")
    print(" Логистическая регрессия в данном случае показала идеальный или почти идеальный результат на этом легком датасете, что часто бывает, когда связи между химическими признаками линейные.")
    
    print("\nВлияние class_weight='balanced':")
    print(" Добавление параметра class_weight='balanced' в RandomForest штрафует деревья за ошибки на редких классах (class_2). Это заставляет алгоритм уделять больше внимания недопредставленным данным, обычно повышая их Recall (или сохраняя его идеальным на легких датасетах). Макро-F1 в таких случаях становится более устойчивым к сдвигам выборки.")
    
    print("\nИтоговый прототип:")
    print(" Для итогового прототипа, если критически важно не пропускать редкие классы (например, дорогой 'class_2'), я выберу LogisticRegression из-за её идеального разделения линейных данных (F1=1.00 во многих случаях на этом датасете). Если взаимосвязи станут сложнее, лучшим выбором будет RandomForestClassifier(class_weight='balanced').")

if __name__ == "__main__":
    main()

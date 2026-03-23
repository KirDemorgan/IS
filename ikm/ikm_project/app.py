"""
app.py — Веб-интерфейс для классификации видов ирисов (Gradio).
"""

import joblib
import numpy as np
import gradio as gr
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

from train import SimpleThresholdClassifier
sys.modules['__main__'].SimpleThresholdClassifier = SimpleThresholdClassifier

CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

model = joblib.load(os.path.join(PROJECT_DIR, 'models', 'best_model.pkl'))
scaler = joblib.load(os.path.join(PROJECT_DIR, 'models', 'scaler.pkl'))


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        result = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(3)}
    else:
        result = {CLASS_NAMES[i]: (1.0 if i == prediction else 0.0) for i in range(3)}

    return result


demo = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Slider(minimum=4.0, maximum=8.0, value=5.1, step=0.1,
                  label="Sepal Length (длина чашелистика, см)"),
        gr.Slider(minimum=2.0, maximum=4.5, value=3.5, step=0.1,
                  label="Sepal Width (ширина чашелистика, см)"),
        gr.Slider(minimum=1.0, maximum=7.0, value=1.4, step=0.1,
                  label="Petal Length (длина лепестка, см)"),
        gr.Slider(minimum=0.1, maximum=2.5, value=0.2, step=0.1,
                  label="Petal Width (ширина лепестка, см)"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Предсказанный вид ириса"),
    title="🌸 Классификация видов ирисов",
    description="Введите измерения цветка и получите предсказание вида ириса.",
    examples=[
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.7, 5.1, 1.6],
        [6.3, 3.3, 6.0, 2.5],
        [5.9, 2.8, 4.8, 1.8],
    ],
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == '__main__':
    demo.launch()

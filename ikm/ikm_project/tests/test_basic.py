"""
test_basic.py — Три автоматических теста для проекта классификации ирисов.

Тест 1: Функция предсказания загружается и работает на корректном примере.
Тест 2: Результат предсказания имеет правильный формат (целое число 0, 1 или 2).
Тест 3: Gradio-приложение создаётся без ошибок.
"""

import joblib
import os
import sys
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from train import SimpleThresholdClassifier
sys.modules['__main__'].SimpleThresholdClassifier = SimpleThresholdClassifier


def load_model_and_scaler():
    model = joblib.load(os.path.join(PROJECT_DIR, 'models', 'best_model.pkl'))
    scaler = joblib.load(os.path.join(PROJECT_DIR, 'models', 'scaler.pkl'))
    return model, scaler


def test_model_loads_and_predicts():
    model, scaler = load_model_and_scaler()
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    assert prediction is not None
    assert len(prediction) == 1


def test_prediction_format():
    model, scaler = load_model_and_scaler()
    samples = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [5.9, 2.8, 4.5, 1.3],
        [6.3, 3.3, 6.0, 2.5],
    ])
    samples_scaled = scaler.transform(samples)
    predictions = model.predict(samples_scaled)
    for pred in predictions:
        assert isinstance(pred, (int, np.integer))
        assert pred in {0, 1, 2}


def test_gradio_app_creates():
    from app import demo
    assert demo is not None
    assert hasattr(demo, 'launch')


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

"""
Тестирование функций plot_series и plot_series_grid из src.utils.visuals_tools.
"""

import numpy as np
import pytest
from src.utils.visuals import plot_series, plot_series_grid

def test_plot_series_single_series():
    """
    Тестирование plot_series с одним временным рядом.
    """
    series = np.sin(np.linspace(0, 10, 100))
    try:
        plot_series(series, plot_title="Тест одного ряда", save_path=None)
    except Exception as e:
        pytest.fail(f"Ошибка при вызове plot_series: {e}")

def test_plot_series_multiple_series():
    """
    Тестирование plot_series с несколькими временными рядами.
    """
    series_list = [
        np.sin(np.linspace(0, 10, 100)),
        np.cos(np.linspace(0, 10, 100))
    ]
    labels = ["sin(t)", "cos(t)"]
    try:
        plot_series(series_list, labels=labels, plot_title="Тест нескольких рядов", save_path=None)
    except Exception as e:
        pytest.fail(f"Ошибка при вызове plot_series: {e}")

def test_plot_series_invalid_labels():
    """
    Тестирование plot_series с некорректным количеством меток.
    """
    series_list = [
        np.sin(np.linspace(0, 10, 100)),
        np.cos(np.linspace(0, 10, 100))
    ]
    labels = ["sin(t)"]  # Недостаточно меток
    with pytest.raises(ValueError):
        plot_series(series_list, labels=labels, plot_title="Некорректные метки", save_path=None)

def test_plot_series_grid():
    """
    Тестирование plot_series_grid.
    """
    series_list = [
        np.sin(np.linspace(0, 10, 100)),
        np.cos(np.linspace(0, 10, 100)),
        np.sin(np.linspace(0, 10, 100)) + np.cos(np.linspace(0, 10, 100))
    ]
    labels = ["sin(t)", "cos(t)", "sin(t) + cos(t)"]
    try:
        plot_series_grid(series_list, labels=labels, plot_title="Тест сетки", save_path=None)
    except Exception as e:
        pytest.fail(f"Ошибка при вызове plot_series_grid: {e}")

def test_plot_series_grid_empty_series():
    """
    Тестирование plot_series_grid с пустым списком рядов.
    """
    with pytest.raises(ValueError):
        plot_series_grid([], plot_title="Пустой список рядов", save_path=None)
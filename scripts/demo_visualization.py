"""
Демонстрация работы функций plot_series и plot_series_grid из src.utils.visuals_tools.
"""

import numpy as np
from src.utils.visuals import plot_series, plot_series_grid

def generate_sample_data():
    """
    Генерация примеров временных рядов.
    """
    t = np.linspace(0, 10, 100)
    series1 = np.sin(t)
    series2 = np.cos(t)
    series3 = np.sin(t) + np.cos(t)
    return [series1, series2, series3]

def demo_plot_series():
    """
    Демонстрация функции plot_series.
    """
    series_list = generate_sample_data()
    labels = ["sin(t)", "cos(t)", "sin(t) + cos(t)"]
    plot_series(
        series_list=series_list,
        labels=labels,
        plot_title="Пример plot_series",
        ylabel="Значение",
        xlabel="Время"
    )

def demo_plot_series_grid():
    """
    Демонстрация функции plot_series_grid.
    """
    series_list = generate_sample_data()
    labels = ["sin(t)", "cos(t)", "sin(t) + cos(t)"]
    plot_series_grid(
        series_list=series_list,
        labels=labels,
        plot_title="Пример plot_series_grid",
        ylabel="Значение",
        xlabel="Время",
        layout='grid',
        figsize=(15, 8)
    )

if __name__ == "__main__":
    print("Демонстрация plot_series:")
    demo_plot_series()

    print("Демонстрация plot_series_grid:")
    demo_plot_series_grid()
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'plot_series',
    'plot_series_grid'
]

def plot_series(series_list, labels=None, plot_title="Заголовок", 
                ylabel="Значение", xlabel="Время", figsize=(12, 6), grid=True):
    """
    Визуализация одного или нескольких временных рядов на одном графике.
    
    Parameters:
    -----------
    series_list : array-like или list
        Один временной ряд или список временных рядов
    labels : str или list, optional
        Одна метка или список меток для каждого ряда
    plot_title : str, optional
        Заголовок графика
    ylabel : str, optional
        Подпись оси Y
    xlabel : str, optional
        Подпись оси X
    figsize : tuple, optional
        Размер фигуры (ширина, высота)
    grid : bool, optional
        Отображать сетку или нет
    """
    # Преобразуем одиночный ряд в список
    if not isinstance(series_list, (list, tuple)):
        series_list = [series_list]
        labels = [labels] if labels else ["Временной ряд"]
    
    # Проверка соответствия количества меток количеству рядов
    if labels and len(labels) != len(series_list):
        raise ValueError("Количество меток должно соответствовать количеству рядов")

    plt.figure(figsize=figsize)
    for i, series in enumerate(series_list):
        label = labels[i] if labels else f"Ряд {i+1}"
        plt.plot(series, label=label)
    
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(grid)
    plt.tight_layout()
    plt.show()


def plot_series_grid(series_list, labels=None, x_series=None, plot_title="Заголовок",
                     ylabel="Значение", xlabel="Время", figsize=(12, 6), grid=True,
                     layout='vertical', nrows=None, ncols=None):
    """
    Визуализация временных рядов в виде сетки графиков.
    
    Parameters:
    -----------
    series_list : list
        Список временных рядов для отображения
    labels : list, optional
        Список меток для каждого ряда
    x_series : array-like, optional
        Значения по оси X
    plot_title : str, optional
        Заголовок графика
    ylabel : str, optional
        Подпись оси Y
    xlabel : str, optional
        Подпись оси X
    figsize : tuple, optional
        Размер фигуры (ширина, высота)
    grid : bool, optional
        Отображать сетку или нет
    layout : str, optional
        'vertical' - графики располагаются в столбец
        'horizontal' - графики располагаются в строку
        'grid' - графики располагаются в виде сетки
    nrows, ncols : int, optional
        Количество строк и столбцов в сетке (только для layout='grid')
    """
    if not series_list:
        raise ValueError("Список временных рядов не может быть пустым")
    
    if labels and len(labels) != len(series_list):
        raise ValueError("Количество меток должно соответствовать количеству рядов")

    n_series = len(series_list)
    
    # Определяем размеры сетки
    if layout == 'vertical':
        nrows, ncols = n_series, 1
    elif layout == 'horizontal':
        nrows, ncols = 1, n_series
    else:  # grid
        if nrows is None and ncols is None:
            ncols = int(np.ceil(np.sqrt(n_series)))
            nrows = int(np.ceil(n_series / ncols))
        elif nrows is None:
            nrows = int(np.ceil(n_series / ncols))
        elif ncols is None:
            ncols = int(np.ceil(n_series / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(nrows, ncols)
    
    for idx, (series, ax) in enumerate(zip(series_list, axes.flat)):
        ax.plot(series)
        ax.set_title(labels[idx] if labels else f"Ряд {idx+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        if x_series is not None:
            step = max(1, len(x_series) // 4)
            ax.set_xticks(range(0, len(x_series), step))
            ax.set_xticklabels(x_series[::step], fontsize=8, rotation=45)
        
        ax.grid(grid)
    
    # Скрываем пустые подграфики
    for ax in axes.flat[len(series_list):]:
        ax.set_visible(False)
    
    fig.suptitle(plot_title)
    plt.tight_layout()
    plt.show()


def main():
    return None

if __name__ == '__main__':
    main()
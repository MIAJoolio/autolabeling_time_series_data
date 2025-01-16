import numpy as np
import matplotlib.pyplot as plt


def single_plot(series, plot_label:str="Временной ряд", plot_title="Заголовок", ylabel:str="Время", xlabel:str="Значение x", figsize=(8,6), grid:bool=True):
    plt.figure(figsize=figsize)
    plt.plot(series, label=plot_label)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(grid)
    plt.show()


def multi_plot_each(series_list, labels=None, plot_title="Заголовок", ylabel="Значение", xlabel="Время", figsize=(12, 6), grid=True):
    """
    Визуализация нескольких временных рядов по горизонтали.
    """
    if labels is not None and len(labels) != len(series_list):
        raise ValueError("Количество меток должно соответствовать количеству рядов.")
    
    fig, axes = plt.subplots(1, len(series_list), figsize=figsize)
    
    if len(series_list) == 1:
        axes = [axes]
    
    for i, series in enumerate(series_list):
        ax = axes[i]
        ax.plot(series, label=labels[i] if labels else None)
        ax.set_title(labels[i] if labels else f"Ряд {i+1}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(grid)
    
    fig.suptitle(plot_title)
    plt.tight_layout()
    plt.show()

def multi_plot_one(series_list, labels=None, plot_title="Заголовок", ylabel="Значение", xlabel="Время", figsize=(12, 6), grid=True):
    """
    Визуализация нескольких временных рядов на одном графике.
    """
    # if labels is not None and len(labels) != len(series_list):
    #     raise ValueError("Количество меток должно соответствовать количеству рядов.")
    
    plt.figure(figsize=figsize)
    
    for i, series in enumerate(series_list):
        
        if labels is None:
            plt.plot(series)
    
        else:
            plt.plot(series, label=labels[i] if labels else f"Ряд {i+1}")
    

    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(grid)
    plt.tight_layout()
    plt.show()

def main():
    return None

if __name__ == '__main__':
    main()
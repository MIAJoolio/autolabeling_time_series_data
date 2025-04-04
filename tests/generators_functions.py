import numpy as np

from core.generation import * 
from core.utils import * 

def show_generators_work():
    """
    Функция по визуализации работы всех генераторов
    """
    # ex 1
    slope = 0.7
    length = 10 

    series = linear_trend(slope, length)
    plot_series_grid(
        series_list=[series], 
        labels=["Временной ряд"], 
        plot_title=f"Линейный тренд {slope=}", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 2
    a = 0.01
    b = 0.1
    c = 2
    length = 10

    series = quadratic_trend(a, b, c, length)
    plot_series_grid(
        series_list=[series], 
        labels=["Временной ряд"], 
        plot_title=f"Квадратичный тренд с параметрами {a=},{b=},{c=}", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 3
    alpha = 0.5
    length = 10

    series = exponential_trend(alpha, length)
    plot_series_grid(
        series_list=[series], 
        labels=["Временной ряд"], 
        plot_title="Экспоненциальный тренд с шумом", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 4
    amplitude = 5.0
    frequency = 2.0
    phase = np.pi / 2
    length = 100

    series = seasonal_series(amplitude, frequency, phase, length)
    plot_series_grid(
        series_list=[series], 
        labels=["Временной ряд"], 
        plot_title="Сезонность (sin/cos) с шумом", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 5
    amplitude = 10.0
    frequency = 2.0
    damping = 0.05
    length = 200

    series = harmonic_oscillator(amplitude, frequency, damping, length)
    plot_series_grid(
        series_list=[series], 
        labels=["Временной ряд"], 
        plot_title="Гармонический осциллятор", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 6
    amplitude = 5.0
    frequency = 4.0
    length = 200

    series = sawtooth_wave(amplitude, frequency, length)
    plot_series_grid(
        series_list=[series], 
        labels=["Временной ряд"], 
        plot_title="Пилообразный сигнал", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

    # ex 7
    initial_value = 10.0
    length = 100
    

    series = random_walk(initial_value, length)
    plot_series_grid(
        series_list=[series], 
        labels=["Временной ряд"], 
        plot_title="Случайное блуждание", 
        xlabel="Время", 
        ylabel="Значение", 
        figsize=(14,3),
        layout='horizontal'
    )

def show_Generator_work():
    # создаем генератор
    generator = Generator(block_length=20)

    # добавляем блоки
    generator.add_block("linear", slope=0.1)
    generator.add_block("seasonal", amplitude=10.0, frequency=1.0, phase=0.0)

    # Генерация временного ряда
    time_series = generator.generate()

    # Вывод результата
    print("Generated time series:", time_series)
    
    # Визуализация результатов
    plot_series_grid(
        [time_series],
        ['full series'],
        plot_title="Сгенерированный временной ряд",
        xlabel="Время",
        ylabel="Значение",
        figsize=(14, 3),
        layout='horizontal'
    )
    
    # Генерация временного ряда
    time_series = generator.generate_with_noise(random_state=42)  
    # Визуализация результатов
    plot_series_grid(
        [time_series],
        ['full series'],
        plot_title="Сгенерированный временной ряд с шумом",
        xlabel="Время",
        ylabel="Значение",
        figsize=(14, 3),
        layout='horizontal'
    )


def main():
    if SHOW_GENS == True:
        show_generators_work()
    
    if SHOW_GENERATOR == True:
        show_Generator_work()

if __name__ == "__main__":

    SHOW_GENS = True
    SHOW_GENERATOR = True

    main()
    
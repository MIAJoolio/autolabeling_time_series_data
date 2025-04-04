from core.generation import *
from core.utils import *  

if __name__ == '__main__':
    
    ts1 = linear_trend(0.5, 100)
    ts2 = linear_trend(-0.5, 100)
    
    plot_series_grid([ts1, ts2], ['восходящий тренд', 'нисходящий тренд'],save_path='/root/autolabeling_time_series_data/tests/trend.png')
    
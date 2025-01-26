import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from io import StringIO
import base64
from sklearn.metrics import silhouette_score
import plotly.colors
from core.feature_extraction.feature_extraction import (
    paa_features, tsa_detrend, tsa_acf, statistical_features,
    signal_peaks_features, stft_features, dft_components,
    dwt_features, dft_signal, dft_approximation
)
from density_clustering import apply_dbscan, apply_birch
from hierarchical_clustering import apply_agglomerative
from partitioning_clustering import apply_kmeans, apply_ts_kmeans

# Инициализация Dash-приложения
app = dash.Dash(__name__, prevent_initial_callbacks='initial_duplicate')

# Макет приложения
app.layout = html.Div([
    # Заголовок для загрузки данных
    html.Label("Загрузка данных"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Перетащите или ', html.A('выберите файл')]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),

    # Заголовок для длины сегмента
    html.Label("Длина сегмента"),
    dcc.Input(
        id='window-size',
        type='number',
        value=4,
        min=1,
        step=1,
        style={'margin': '10px'}
    ),

    # Галочки для выбора типа кластеризации
    dcc.Checklist(
        id='clustering-type-checkbox',
        options=[
            {'label': 'Свободный режим', 'value': 'free'},
            {'label': 'Кластеры различимы', 'value': 'distinct'},
            {'label': 'Скученные кластеры', 'value': 'dense'},
            {'label': 'Ложные кластеры', 'value': 'false'}
        ],
        value=[],
        style={'margin': '10px'}
    ),

    # Скрытые элементы для типа предобработки
    html.Div(
        id='preprocessing-div',
        children=[
            html.Label("Тип предобработки"),
            dcc.Dropdown(
                id='method-selector',
                options=[
                    {'label': 'Raw', 'value': 'raw'},
                    {'label': 'PAA', 'value': 'paa_features'},
                    {'label': 'Detrend', 'value': 'tsa_detrend'},
                    {'label': 'ACF', 'value': 'tsa_acf'},
                    {'label': 'Statistical Features', 'value': 'statistical_features'},
                    {'label': 'Signal Peaks Features', 'value': 'signal_peaks_features'},
                    {'label': 'STFT Features', 'value': 'stft_features'},
                    {'label': 'DFT Components', 'value': 'dft_components'},
                    {'label': 'DWT Features', 'value': 'dwt_features'},
                    {'label': 'DFT Signal', 'value': 'dft_signal'},
                    {'label': 'DFT Approximation', 'value': 'dft_approximation'}
                ],
                value='raw',
                placeholder="Выберите метод",
                style={'margin': '10px'}
            )
        ],
        style={'display': 'none'}  # Скрываем по умолчанию
    ),

    # Скрытые элементы для алгоритма кластеризации
    html.Div(
        id='clustering-div',
        children=[
            html.Label("Алгоритм кластеризации"),
            dcc.Dropdown(
                id='clustering-method-selector',
                options=[
                    {'label': 'DBSCAN', 'value': 'dbscan'},
                    {'label': 'BIRCH', 'value': 'birch'},
                    {'label': 'Agglomerative', 'value': 'agglomerative'},
                    {'label': 'KMeans', 'value': 'kmeans'},
                    {'label': 'Time Series KMeans', 'value': 'ts_kmeans'}
                ],
                value='dbscan',
                placeholder="Выберите метод кластеризации",
                style={'margin': '10px'}
            )
        ],
        style={'display': 'none'}  # Скрываем по умолчанию
    ),

    # Ползунок для параметра кластеризации
    html.Label("Расстояние между кластерами (cluster_distance)"),
    dcc.Slider(
        id='cluster-distance-slider',
        min=0,
        max=1,
        step=0.1,
        value=0.5,
        marks={i: str(i) for i in np.arange(0, 1.1, 0.1)},
        tooltip={"placement": "bottom", "always_visible": True},
        className="slider-style"  # Используем className для стилизации
    ),

    # Новый ползунок для выбора значения в диапазоне данных
    html.Label("Выберите значение в диапазоне данных"),
    dcc.Slider(
        id='data-range-slider',
        min=0,
        max=1,  # Временно, будет обновлено после загрузки данных
        step=0.1,
        value=0.5,
        marks={},
        tooltip={"placement": "bottom", "always_visible": True},
        className="slider-style"
    ),

    # Таблица для отображения результатов silhouette score
    dash_table.DataTable(
        id='silhouette-table',
        columns=[
            {'name': 'distance_threshold', 'id': 'distance_threshold'},
            {'name': 'Silhouette Score', 'id': 'silhouette_score'}
        ],
        data=[],
        style_table={'margin': '10px'},
        style_cell={'textAlign': 'left'},
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    ),

    # Кнопки
    html.Button('Показать сегменты', id='show-segments-button', style={'margin': '10px'}),
    html.Button('Вернуться к исходному ряду', id='reset-button', style={'margin': '10px'}),
    html.Button('Применить метод', id='apply-method-button', style={'margin': '10px'}),
    html.Button('Кластеризовать', id='cluster-button', style={'margin': '10px'}),

    # График
    dcc.Graph(id='graph'),

    # Хранилища данных
    dcc.Store(id='stored-data', data={'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()}),
    dcc.Store(id='stored-segments', data=[]),
    dcc.Store(id='transformed-segments', data=[]),
    dcc.Store(id='cluster-labels', data=[])
])

# Callback для загрузки данных и обновления ползунка
@app.callback(
    Output('stored-data', 'data'),
    Output('data-range-slider', 'min'),
    Output('data-range-slider', 'max'),
    Output('data-range-slider', 'value'),
    Output('data-range-slider', 'marks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(contents, filename):
    if contents is None:
        default_data = {'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()}
        return default_data, 0, 1, 0.5, {}

    content_type, content_string = contents.split(',')
    decoded = StringIO(base64.b64decode(content_string).decode('utf-8', errors='ignore'))

    try:
        df = pd.read_csv(decoded)
        if len(df) > 10000:
            df = df.iloc[:10000]
        if 'x' in df.columns and 'y' in df.columns:
            data = {'x': df['x'].tolist(), 'y': df['y'].tolist()}
        else:
            data = {'x': df.iloc[:, 0].tolist(), 'y': df.iloc[:, 1].tolist()}

        # Определяем минимальное и максимальное значение в данных
        y_values = data['y']
        min_value = min(y_values)
        max_value = max(y_values)

        # Создаем метки для ползунка
        marks = {i: f"{i:.1f}" for i in np.linspace(min_value, max_value, 5)}

        return data, min_value, max_value, (min_value + max_value) / 2, marks

    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        default_data = {'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()}
        return default_data, 0, 1, 0.5, {}

# Callback для создания сегментов
@app.callback(
    Output('stored-segments', 'data'),
    Input('show-segments-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('window-size', 'value'),
    prevent_initial_call=True
)
def create_segments(n_clicks, data, window_size):
    if n_clicks is None:
        return []
    
    y = np.array(data['y'])
    segments = [y[i:i + window_size].tolist() for i in range(0, len(y), window_size) if len(y[i:i + window_size]) == window_size]
    return segments

# Callback для отображения/скрытия элементов интерфейса
@app.callback(
    Output('preprocessing-div', 'style'),
    Output('clustering-div', 'style'),
    Input('clustering-type-checkbox', 'value')
)
def toggle_interface(clustering_type):
    if 'free' in clustering_type:  # Если выбран свободный режим
        return {'display': 'block'}, {'display': 'block'}
    else:  # Если свободный режим не выбран
        return {'display': 'none'}, {'display': 'none'}

# Callback для применения предобработки к сегментам
@app.callback(
    Output('transformed-segments', 'data'),
    Input('apply-method-button', 'n_clicks'),
    State('stored-segments', 'data'),
    State('method-selector', 'value'),
    prevent_initial_call=True
)
def apply_preprocessing(n_clicks, segments, method):
    if n_clicks is None or not segments:
        return []
    
    transformed_segments = []
    for segment in segments:
        if method == 'raw':
            transformed_segments.append(segment)
        elif method == 'paa_features':
            transformed_segments.append(paa_features(np.array(segment)))
        elif method == 'tsa_detrend':
            transformed_segments.append(tsa_detrend(np.array(segment)))
        elif method == 'tsa_acf':
            transformed_segments.append(tsa_acf(np.array(segment)))
        elif method == 'statistical_features':
            transformed_segments.append(statistical_features(np.array(segment)))
        elif method == 'signal_peaks_features':
            transformed_segments.append(signal_peaks_features(np.array(segment)))
        elif method == 'stft_features':
            transformed_segments.append(stft_features(np.array(segment)))
        elif method == 'dft_components':
            transformed_segments.append(dft_components(np.array(segment)))
        elif method == 'dwt_features':
            transformed_segments.append(dwt_features(np.array(segment)))
        elif method == 'dft_signal':
            transformed_segments.append(dft_signal(np.array(segment)))
        elif method == 'dft_approximation':
            transformed_segments.append(dft_approximation(np.array(segment)))
    
    return transformed_segments

# Callback для обновления параметров кластеризации и таблицы silhouette score
@app.callback(
    Output('cluster-distance-slider', 'min'),
    Output('cluster-distance-slider', 'max'),
    Output('cluster-distance-slider', 'value'),
    Output('silhouette-table', 'data'),
    Input('stored-data', 'data'),
    Input('clustering-type-checkbox', 'value'),
    State('stored-segments', 'data'),
    State('transformed-segments', 'data'),
    State('clustering-method-selector', 'value'),
    prevent_initial_call=True
)
def update_clustering_params(data, clustering_type, segments, transformed_segments, clustering_method):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, []
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not data or 'free' not in clustering_type:
        return dash.no_update, dash.no_update, dash.no_update, []
    
    y = np.array(data['y'])
    max_value = np.max(y)  # Максимальное значение в данных
    
    if triggered_id == 'stored-data':
        return 1, max_value, max_value / 2, []
    
    data_to_cluster = transformed_segments if transformed_segments else segments
    if not data_to_cluster:
        return 1, max_value, max_value / 2, []
    
    data_array = np.array(data_to_cluster)
    results = []
    best_score = -1
    best_threshold = max_value / 2  # Начальное значение ползунка

    for threshold in np.arange(1, max_value + 1, (max_value - 1) / 10):  # 10 шагов
        if clustering_method == 'dbscan':
            labels, _ = apply_dbscan(data_array, eps=threshold, min_samples=5)
        elif clustering_method == 'birch':
            labels, _ = apply_birch(data_array, threshold=threshold, n_clusters=None)
        elif clustering_method == 'agglomerative':
            labels, _ = apply_agglomerative(data_array, distance_threshold=threshold, n_clusters=None)
        elif clustering_method == 'kmeans':
            labels, _ = apply_kmeans(data_array, n_clusters=int(threshold))
        elif clustering_method == 'ts_kmeans':
            labels, _ = apply_ts_kmeans(data_array, n_clusters=int(threshold))
        else:
            labels = []
        
        if len(set(labels)) > 1:  # Silhouette score требует как минимум 2 кластера
            score = silhouette_score(data_array, labels)
            results.append({'distance_threshold': round(threshold, 1), 'silhouette_score': round(score, 3)})
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    return 1, max_value, best_threshold, results

# Callback для кластеризации
@app.callback(
    Output('cluster-labels', 'data'),
    Input('cluster-button', 'n_clicks'),
    State('stored-segments', 'data'),
    State('transformed-segments', 'data'),
    State('cluster-distance-slider', 'value'),
    State('clustering-method-selector', 'value'),
    prevent_initial_call=True
)
def apply_clustering(n_clicks, segments, transformed_segments, cluster_distance, clustering_method):
    if n_clicks is None:
        return []
    
    data_to_cluster = transformed_segments if transformed_segments else segments
    if not data_to_cluster:
        return []
    
    data_array = np.array(data_to_cluster)
    
    if clustering_method == 'dbscan':
        labels, _ = apply_dbscan(data_array, eps=cluster_distance, min_samples=5)
    elif clustering_method == 'birch':
        labels, _ = apply_birch(data_array, threshold=cluster_distance, n_clusters=None)
    elif clustering_method == 'agglomerative':
        labels, _ = apply_agglomerative(data_array, distance_threshold=cluster_distance, n_clusters=None)
    elif clustering_method == 'kmeans':
        labels, _ = apply_kmeans(data_array, n_clusters=int(cluster_distance))
    elif clustering_method == 'ts_kmeans':
        labels, _ = apply_ts_kmeans(data_array, n_clusters=int(cluster_distance))
    else:
        labels = []
    
    return labels.tolist()

# Общий callback для обновления графика
@app.callback(
    Output('graph', 'figure'),
    Input('show-segments-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('apply-method-button', 'n_clicks'),
    Input('cluster-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('stored-segments', 'data'),
    State('transformed-segments', 'data'),
    State('cluster-labels', 'data'),
    State('window-size', 'value'),
    State('method-selector', 'value'),
    prevent_initial_call=True
)
def update_graph(show_clicks, reset_clicks, apply_method_clicks, cluster_clicks, data, segments, transformed_segments, cluster_labels, window_size, method):
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    x = data['x']
    y = np.array(data['y'])
    
    if button_id == 'reset-button':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Исходный график'))
        
        for i in range(len(segments)):
            x_line = i * window_size
            fig.add_trace(go.Scatter(
                x=[x_line, x_line],
                y=[min(y), max(y)],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
        
        for i in range(len(segments)):
            x_text = i * window_size + window_size / 2
            y_text = max(y)
            fig.add_trace(go.Scatter(
                x=[x_text],
                y=[y_text],
                mode='text',
                text=[f'Сегмент {i + 1}'],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Исходный график с границами сегментов",
            xaxis_title="Индекс",
            yaxis_title="Значение",
            showlegend=True
        )
        return fig
    
    elif button_id == 'show-segments-button':
        fig = go.Figure()
        for i, segment in enumerate(segments):
            x_seg = np.arange(len(segment)) + i * window_size
            fig.add_trace(go.Scatter(
                x=x_seg,
                y=segment,
                mode='lines+markers',
                name=f'Сегмент {i + 1}'
            ))
        
        fig.update_layout(
            title="Сегменты данных",
            xaxis_title="Индекс",
            yaxis_title="Значение"
        )
        return fig
    
    elif button_id == 'apply-method-button':
        fig = go.Figure()
        for i, segment in enumerate(transformed_segments):
            x_seg = np.arange(len(segment)) + i * window_size
            fig.add_trace(go.Scatter(
                x=x_seg,
                y=segment,
                mode='lines+markers',
                name=f'Сегмент {i + 1} ({method})'
            ))
        
        fig.update_layout(
            title=f"Результат для сегментов ({method})",
            xaxis_title="Индекс",
            yaxis_title="Значение"
        )
        return fig
    
    elif button_id == 'cluster-button':
        fig = go.Figure()
        if not segments or not cluster_labels:
            return go.Figure()
        
        # Используем исходные сегменты для отображения, но метки кластеров из обработанных данных
        data_to_plot = segments
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels)  # Количество уникальных меток кластеров
        
        # Расширяем цветовую палитру, если кластеров больше, чем цветов в Plotly
        colors = plotly.colors.qualitative.Plotly
        if n_clusters > len(colors):
            colors = colors * (n_clusters // len(colors) + 1)  # Повторяем палитру
        
        # Отрисовка сегментов с метками кластеров
        for i, segment in enumerate(data_to_plot):
            if i >= len(cluster_labels):  # Проверяем, что индекс не выходит за пределы списка меток
                break
            cluster_label = cluster_labels[i]
            
            # Если метка кластера -1 (шум), используем серый цвет
            if cluster_label == -1:
                color = 'gray'
            else:
                color = colors[cluster_label % len(colors)]  # Используем модуль для избежания IndexError
            
            x_seg = np.arange(len(segment)) + i * window_size
            fig.add_trace(go.Scatter(
                x=x_seg,
                y=segment,
                mode='lines+markers',
                name=f'Кластер {cluster_label}',
                marker=dict(color=color),
                line=dict(color=color)
            ))
        
        # Добавляем вертикальные линии для границ сегментов
        for i in range(len(segments)):
            x_line = i * window_size
            fig.add_trace(go.Scatter(
                x=[x_line, x_line],
                y=[min(y), max(y)],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
        
        # Добавляем текстовые аннотации для номеров сегментов
        for i in range(len(segments)):
            x_text = i * window_size + window_size / 2  # Центр сегмента
            y_text = max(y)  # Размещаем текст сверху
            fig.add_trace(go.Scatter(
                x=[x_text],
                y=[y_text],
                mode='text',
                text=[f'Сегмент {i + 1}'],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Кластеризация сегментов (исходные данные с метками кластеров)",
            xaxis_title="Индекс",
            yaxis_title="Значение",
            showlegend=True  # Включаем легенду
        )
        return fig

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)
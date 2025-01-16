import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from io import StringIO
import base64
from sklearn.cluster import KMeans
import plotly.colors

# Импорт функций из feature_extraction
from feature_extraction import (
    paa_features, tsa_detrend, tsa_acf, statistical_features,
    signal_peaks_features, stft_features, dft_components,
    dwt_features, mean_dwt_features, dwt_features_with_info
)

# Инициализация Dash-приложения
app = dash.Dash(__name__)

# Макет приложения
app.layout = html.Div([
    # 1. Панель загрузки файла
    html.Div([
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
    ]),

    # 2. Поле выбора размера окна и кнопки
    html.Div([
        dcc.Input(id='window-size', type='number', value=4, min=1, step=1, placeholder="Размер окна"),
        html.Button('Вернуться к исходному ряду', id='reset-button'),
        html.Button('Показать фрагменты', id='show-fragments-button'),
    ], style={'margin': '10px'}),

    # 3. Выбор метода предобработки и кнопка "Применить метод"
    html.Div([
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
                {'label': 'Mean DWT Features', 'value': 'mean_dwt_features'},
                {'label': 'DWT Features with Info', 'value': 'dwt_features_with_info'}
            ],
            value='raw',  # По умолчанию выбран метод Raw
            placeholder="Выберите метод"
        ),
        html.Button('Применить метод', id='apply-method-button'),
    ], style={'margin': '10px'}),

    # 4. KMeans с выбором числа кластеров и кнопка "Кластеризовать"
    html.Div([
        dcc.Input(id='n-clusters', type='number', value=3, min=1, step=1, placeholder="Число кластеров"),
        html.Button('Кластеризовать', id='cluster-button'),
    ], style={'margin': '10px'}),

    # 5. Кнопка для сброса индексов
    html.Div([
        html.Button('Сбросить индексы', id='reset-indices-button'),
    ], style={'margin': '10px'}),

    # График
    dcc.Graph(id='graph'),

    # Хранение данных
    dcc.Store(id='stored-data', data={'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()}),
    dcc.Store(id='stored-fragments', data=[]),  # Исходные фрагменты
    dcc.Store(id='transformed-fragments', data=[]),  # Преобразованные фрагменты
    dcc.Store(id='cluster-labels', data=[]),  # Метки кластеров
    dcc.Store(id='reset-indices-flag', data=False),  # Флаг для сброса индексов
])

# Callback для загрузки данных
@app.callback(
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def load_data(contents, filename):
    if contents is None:
        return {'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()}
    
    content_type, content_string = contents.split(',')
    decoded = StringIO(base64.b64decode(content_string).decode('utf-8', errors='ignore'))
    
    try:
        df = pd.read_csv(decoded)
        if len(df) > 10000:
            df = df.iloc[:10000]
        if 'x' in df.columns and 'y' in df.columns:
            return {'x': df['x'].tolist(), 'y': df['y'].tolist()}
        else:
            return {'x': df.iloc[:, 0].tolist(), 'y': df.iloc[:, 1].tolist()}
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return {'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()}

# Callback для создания фрагментов
@app.callback(
    Output('stored-fragments', 'data'),
    Input('show-fragments-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('window-size', 'value'),
    prevent_initial_call=True
)
def create_fragments(n_clicks, data, window_size):
    if n_clicks is None:
        return []
    
    y = np.array(data['y'])
    fragments = [y[i:i + window_size].tolist() for i in range(0, len(y), window_size)]
    return fragments

# Callback для применения метода преобразования
@app.callback(
    Output('transformed-fragments', 'data'),
    Input('apply-method-button', 'n_clicks'),
    State('stored-fragments', 'data'),
    State('window-size', 'value'),
    State('method-selector', 'value'),
    prevent_initial_call=True
)
def apply_method(n_clicks, fragments, window_size, method):
    if n_clicks is None:
        return []
    
    transformed_fragments = []
    for fragment in fragments:
        if method == 'raw':
            result = fragment
        elif method == 'paa_features':
            result = paa_features(np.array(fragment), n_segments=window_size)
        elif method == 'tsa_detrend':
            result = tsa_detrend(np.array(fragment))
        elif method == 'tsa_acf':
            result = tsa_acf(np.array(fragment), n_lags=window_size)
        elif method == 'statistical_features':
            result = statistical_features(np.array(fragment))
        elif method == 'signal_peaks_features':
            result = signal_peaks_features(np.array(fragment))
        elif method == 'stft_features':
            result = stft_features(np.array(fragment))
        elif method == 'dft_components':
            result = dft_components(np.array(fragment), n_freqs=window_size)
        elif method == 'dwt_features':
            result = dwt_features(np.array(fragment), level=window_size)
        elif method == 'mean_dwt_features':
            result = mean_dwt_features(np.array(fragment), level=window_size)
        elif method == 'dwt_features_with_info':
            result = dwt_features_with_info(np.array(fragment), level=window_size)
        else:
            result = fragment
        
        transformed_fragments.append(result)
    
    return transformed_fragments

# Callback для сброса индексов
@app.callback(
    Output('reset-indices-flag', 'data'),
    Input('reset-indices-button', 'n_clicks'),
    State('reset-indices-flag', 'data'),
    prevent_initial_call=True
)
def toggle_reset_indices(n_clicks, reset_flag):
    if n_clicks is None:
        return reset_flag
    return not reset_flag  # Инвертируем флаг

# Callback для кластеризации
@app.callback(
    Output('cluster-labels', 'data'),
    Input('cluster-button', 'n_clicks'),
    State('stored-fragments', 'data'),
    State('transformed-fragments', 'data'),
    State('n-clusters', 'value'),
    prevent_initial_call=True
)
def apply_clustering(n_clicks, fragments, transformed_fragments, n_clusters):
    if n_clicks is None:
        return []
    
    # Используем преобразованные данные, если они есть, иначе исходные фрагменты
    data_to_cluster = transformed_fragments if transformed_fragments else fragments
    
    if not data_to_cluster or n_clusters > len(data_to_cluster):
        return []  # Если данных нет или кластеров больше, чем фрагментов
    
    # Проверяем, что все фрагменты имеют одинаковую длину
    lengths = [len(fragment) for fragment in data_to_cluster]
    if len(set(lengths)) != 1:
        print("Ошибка: фрагменты имеют разную длину. Кластеризация невозможна.")
        return []
    
    data_array = np.array(data_to_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_array)
    return labels.tolist()

# Callback для обновления графика
@app.callback(
    Output('graph', 'figure'),
    Input('show-fragments-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('apply-method-button', 'n_clicks'),
    Input('cluster-button', 'n_clicks'),
    Input('reset-indices-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('stored-fragments', 'data'),
    State('transformed-fragments', 'data'),
    State('window-size', 'value'),
    State('method-selector', 'value'),
    State('cluster-labels', 'data'),
    State('reset-indices-flag', 'data'),
    prevent_initial_call=True
)
def update_graph(show_clicks, reset_clicks, apply_method_clicks, cluster_clicks, reset_indices_clicks, data, fragments, transformed_fragments, window_size, method, cluster_labels, reset_indices_flag):
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    x = data['x']
    y = np.array(data['y'])
    
    if button_id == 'reset-button':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Исходный график'))
        return fig
    
    elif button_id == 'show-fragments-button':
        fig = go.Figure()
        for i, fragment in enumerate(fragments):
            x_frag = np.arange(len(fragment)) + i * window_size
            fig.add_trace(go.Scatter(
                x=x_frag,
                y=fragment,
                mode='lines+markers',
                name=f'Фрагмент {i + 1}'
            ))
        
        # Добавляем вертикальные линии и подписи
        for i in range(len(fragments)):
            x_line = i * window_size
            fig.add_trace(go.Scatter(
                x=[x_line, x_line],
                y=[min(y), max(y)],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
            x_text = x_line + window_size / 2
            fig.add_trace(go.Scatter(
                x=[x_text],
                y=[max(y)],
                mode='text',
                text=[f'Фрагмент {i + 1}'],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Фрагменты данных",
            xaxis_title="Индекс",
            yaxis_title="Значение"
        )
        return fig
    
    elif button_id == 'apply-method-button':
        fig = go.Figure()
        for i, fragment in enumerate(transformed_fragments):
            if reset_indices_flag:
                x_frag = np.arange(len(fragment))  # Сбрасываем индексы
            else:
                x_frag = np.arange(len(fragment)) + i * window_size
            fig.add_trace(go.Scatter(
                x=x_frag,
                y=fragment,
                mode='lines+markers',
                name=f'Фрагмент {i + 1} ({method})'
            ))
        
        # Добавляем вертикальные линии и подписи
        for i in range(len(transformed_fragments)):
            if reset_indices_flag:
                x_line = i * window_size
            else:
                x_line = i * window_size
            fig.add_trace(go.Scatter(
                x=[x_line, x_line],
                y=[min(y), max(y)],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
            x_text = x_line + window_size / 2
            fig.add_trace(go.Scatter(
                x=[x_text],
                y=[max(y)],
                mode='text',
                text=[f'Фрагмент {i + 1}'],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Результат для фрагментов ({method})",
            xaxis_title="Индекс",
            yaxis_title="Значение"
        )
        return fig
    
    elif button_id == 'cluster-button':
        fig = go.Figure()
        if not fragments or not cluster_labels:
            return go.Figure()
        
        # Используем исходные фрагменты для отображения
        data_to_plot = fragments
        
        # Создаем цветовую палитру
        n_clusters = max(cluster_labels) + 1
        colors = plotly.colors.qualitative.Plotly[:n_clusters]
        
        for i, fragment in enumerate(data_to_plot):
            if i >= len(cluster_labels):
                break
            cluster_label = cluster_labels[i]
            color = colors[cluster_label]
            if reset_indices_flag:
                x_frag = np.arange(len(fragment))  # Сбрасываем индексы
            else:
                x_frag = np.arange(len(fragment)) + i * window_size
            fig.add_trace(go.Scatter(
                x=x_frag,
                y=fragment,
                mode='lines+markers',
                name=f'Кластер {cluster_label}',
                marker=dict(color=color),
                line=dict(color=color)
            ))
        
        # Добавляем вертикальные линии и подписи
        for i in range(len(fragments)):
            if reset_indices_flag:
                x_line = i * window_size
            else:
                x_line = i * window_size
            fig.add_trace(go.Scatter(
                x=[x_line, x_line],
                y=[min(y), max(y)],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
            x_text = x_line + window_size / 2
            fig.add_trace(go.Scatter(
                x=[x_text],
                y=[max(y)],
                mode='text',
                text=[f'Фрагмент {i + 1}'],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Кластеризация фрагментов",
            xaxis_title="Индекс",
            yaxis_title="Значение",
            showlegend=True
        )
        return fig

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)
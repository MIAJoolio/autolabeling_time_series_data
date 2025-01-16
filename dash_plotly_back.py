import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from io import StringIO
import base64
from sklearn.cluster import KMeans
import plotly.colors
from feature_extraction import (
    paa_features, tsa_detrend, tsa_acf, statistical_features,
    signal_peaks_features, stft_features, dft_components,
    dwt_features, mean_dwt_features, dwt_features_with_info
)

# Инициализация Dash-приложения
app = dash.Dash(__name__)

# Макет приложения
app.layout = html.Div([
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
    dcc.Input(id='window-size', type='number', value=4, min=1, step=1),
    dcc.Input(id='n-clusters', type='number', value=3, min=1, step=1, placeholder="Число кластеров"),
    dcc.Dropdown(
        id='method-selector',
        options=[
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
        value='paa_features',
        placeholder="Выберите метод"
    ),
    html.Button('Показать фрагменты', id='show-fragments-button'),
    html.Button('Вернуться к исходному ряду', id='reset-button'),
    html.Button('Применить метод', id='apply-method-button'),
    html.Button('Кластеризовать', id='cluster-button'),
    dcc.Graph(id='graph'),
    dcc.Store(id='stored-data', data={'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()}),
    dcc.Store(id='stored-fragments', data=[]),  # Исходные фрагменты
    dcc.Store(id='transformed-fragments', data=[]),  # Преобразованные фрагменты
    dcc.Store(id='cluster-labels', data=[])  # Метки кластеров
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
    fragments = [y[i:i + window_size].tolist() for i in range(0, len(y), window_size) if len(y[i:i + window_size]) == window_size]
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
        if method == 'paa_features':
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
        
        transformed_fragments.append(result.tolist())
    
    return transformed_fragments

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
    
    if not data_to_cluster:  # Если данных нет, возвращаем пустой список
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
    State('stored-data', 'data'),
    State('stored-fragments', 'data'),
    State('transformed-fragments', 'data'),
    State('window-size', 'value'),
    State('method-selector', 'value'),
    State('cluster-labels', 'data'),
    prevent_initial_call=True
)
def update_graph(show_clicks, reset_clicks, apply_method_clicks, cluster_clicks, data, fragments, transformed_fragments, window_size, method, cluster_labels):
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
            x_frag = np.arange(len(fragment))
            fig.add_trace(go.Scatter(
                x=x_frag,
                y=fragment,
                mode='lines+markers',
                name=f'Фрагмент {i + 1}'
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
            fig.add_trace(go.Scatter(
                x=np.arange(len(fragment)),
                y=fragment,
                mode='lines+markers',
                name=f'Фрагмент {i + 1} ({method})'
            ))
        
        fig.update_layout(
            title=f"Результат для фрагментов ({method})",
            xaxis_title="Индекс",
            yaxis_title="Значение"
        )
        return fig
    
    elif button_id == 'cluster-button':
        fig = go.Figure()
        if not fragments or not cluster_labels:  # Проверяем, что фрагменты и метки кластеров существуют
            return go.Figure()
        
        # Используем исходные фрагменты для отображения
        data_to_plot = fragments
        
        # Создаем цветовую палитру
        n_clusters = max(cluster_labels) + 1  # Количество кластеров
        colors = plotly.colors.qualitative.Plotly[:n_clusters]  # Используем палитру Plotly
        
        # Отрисовка фрагментов с метками кластеров
        for i, fragment in enumerate(data_to_plot):
            if i >= len(cluster_labels):  # Проверяем, что индекс не выходит за пределы списка меток
                break
            cluster_label = cluster_labels[i]
            color = colors[cluster_label]  # Получаем цвет для текущего кластера
            x_frag = np.arange(len(fragment)) + i * window_size
            fig.add_trace(go.Scatter(
                x=x_frag,
                y=fragment,
                mode='lines+markers',
                name=f'Кластер {cluster_label}',
                marker=dict(color=color),
                line=dict(color=color)
            ))
        
        # Добавляем вертикальные линии для границ фрагментов
        for i in range(len(fragments)):
            x_line = i * window_size
            fig.add_trace(go.Scatter(
                x=[x_line, x_line],
                y=[min(y), max(y)],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ))
        
        # Добавляем текстовые аннотации для номеров фрагментов
        for i in range(len(fragments)):
            x_text = i * window_size + window_size / 2  # Центр фрагмента
            y_text = max(y)  # Размещаем текст сверху
            fig.add_trace(go.Scatter(
                x=[x_text],
                y=[y_text],
                mode='text',
                text=[f'Фрагмент {i + 1}'],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Кластеризация фрагментов (исходные данные)",
            xaxis_title="Индекс",
            yaxis_title="Значение",
            showlegend=True  # Включаем легенду
        )
        return fig

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)
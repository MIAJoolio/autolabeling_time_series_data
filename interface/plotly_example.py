import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from io import StringIO
import base64

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
    html.Button('Показать фрагменты', id='show-fragments-button'),
    html.Button('Вернуться к исходному ряду', id='reset-button'),
    dcc.Graph(id='graph'),
    dcc.Store(id='stored-data', data={'x': np.arange(16).tolist(), 'y': np.random.rand(16).tolist()})
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

# Callback для обновления графика
@app.callback(
    Output('graph', 'figure'),
    Input('show-fragments-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    State('stored-data', 'data'),
    State('window-size', 'value'),
    prevent_initial_call=True
)
def update_graph(show_clicks, reset_clicks, data, window_size):
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    x = data['x']
    y = data['y']
    
    if button_id == 'reset-button':
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Исходный график'))
        return fig
    
    elif button_id == 'show-fragments-button':
        def split_data(data, window):
            return [data[i:i + window] for i in range(0, len(data), window)]
        
        y_fragments = split_data(y, window_size)[:10]  # Ограничиваем количество фрагментов
        
        fig = go.Figure()
        for i, y_frag in enumerate(y_fragments):
            x_frag = np.arange(len(y_frag))
            fig.add_trace(go.Scatter(
                x=x_frag,
                y=y_frag,
                mode='lines+markers',
                name=f'Фрагмент {i + 1}'
            ))
        
        fig.update_layout(
            title="Фрагменты данных",
            xaxis_title="Индекс",
            yaxis_title="Значение"
        )
        return fig

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)
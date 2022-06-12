import dash
import dash_bootstrap_components as dbc

from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State


from sklearn.preprocessing import MinMaxScaler

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.serialize import model_to_json, model_from_json

import requests
import pandas as pd
import numpy as np

import datetime as dt
import time

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model


"""  LOAD DATA  """

data_or = pd.read_csv('data_hack.csv',
                      on_bad_lines='skip',
                      sep=';',
                      parse_dates=['create_date'],
                      infer_datetime_format=True
                      )
df = data_or[data_or['create_date'] >= '01-01-2017']
df = df.drop_duplicates()
df = df.dropna(how='any')
df['avg_bill'] = df['avg_bill']/70
df['create_date'] = pd.to_datetime(df['create_date'])



"""  START DASH  """

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP]
                )



'''  SLIDERS  '''

# создаем слайдер для типа канала
channel_type_selector = dcc.Dropdown(
    id='channel-selector',
    options=df['type_channel'].unique(),
    value=df['type_channel'].unique(),
    multi=True
)


time_selector = dcc.RangeSlider(
    id='time-slider',
    min=time.mktime(min(df['create_date']).timetuple()),
    max=time.mktime(max(df['create_date']).timetuple()),
    marks={
           int(time.mktime(min(df['create_date']).timetuple())): '2017',
           int(time.mktime(dt.date(2018, 1, 1).timetuple())): '2018',
           int(time.mktime(dt.date(2019, 1, 1).timetuple())): '2019',
           int(time.mktime(dt.date(2020, 1, 1).timetuple())): '2020',
           int(time.mktime(dt.date(2021, 1, 1).timetuple())): '2021',
           int(time.mktime(dt.date(2022, 1, 1).timetuple())): '2022',
           int(time.mktime(max(df['create_date']).timetuple())): 'Н.В.'
           },
    step=1,
    value=[time.mktime(min(df['create_date']).timetuple()),
           time.mktime(max(df['create_date']).timetuple())]
)

# fig = px.scatter(df, x='type_channel', y='click_price')







'''  GLOBAL DESIGN SETTINGS  '''

CHARTS_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(
            family='Century Gothic',
            size=14
        ),
        legend=dict(orientation='h',
                    title_text='',
                    x=0,
                    y=1.1)
    )
)

"""  TABS  """

tab1_content = dbc.Row([

                dbc.Col([
                ], id='head-TAG', width={'size': 12}),

                dcc.Store(id='filtered-data', storage_type='session'),

                # LEFT main col
                dbc.Col([

                    dbc.Row([
                        # показатель
                        dbc.Col([
                            html.Div(['ROMI = (ДОХОДЫ-РАСХОДЫ НА МАРКЕТИНГ)/РАСХОДЫ НА МАРКЕТИНГ * 100%'], className='title-romi'),
                            html.Div([], className='border'),
                            html.Div(id='romi', className='number_romi')
                        ], className='metr-tomi', width={'size': 4}),

                        dbc.Col([dcc.Graph(id='pie-chart')
                        ], width={'size': 8})
                    ], className='metr-left', justify='between'),


                    # селектор
                    dbc.Row([
                        dbc.Col([
                            html.Div('ВЫБОР КАНАЛОВ ПРОДВИЖЕНИЯ'),
                            # селектор времени
                            html.Div(channel_type_selector)
                        ], className='selector-time', width={'size': 12})
                    ], className='selector-box'),

                    # графики
                    dbc.Row([
                        dbc.Col([dcc.Graph(id='dist-temp-chart')
                        ], className='chart-time', width={'size': 12}),
                    ], className='margin-rows'),

                    dbc.Row([
                        dbc.Col(['ЗАТРАТЫ НА ПРИВЛЕЧЕНИЕ/КОЛИЧЕСТВО НОВЫХ КЛИЕНТОВ'
                        ], className='chart-time', width={'size': 12})
                    ], className='margin-rows')
                ], className='main-col-left', width={'size': 5}),

                # RIGHT main col
                dbc.Col([

                    # показатели
                    dbc.Row([
                        dbc.Col([
                            html.Div(['СРЕДНИЙ ЧЕК ЗА ПЕРИОД'], className='title-box color-yellow'),
                            html.Div([], className='border'),
                            html.Div(id='avg-bill', className='number')
                        ], className='metr-stand', width={'size': 3}),

                        dbc.Col([
                            html.Div(['КОНВЕРСИЯ ЗА ПЕРИОД'], className='title-box color-green'),
                            html.Div([], className='border'),
                            html.Div(id='conv', className='number')
                        ], className='metr-stand', width={'size': 3}),

                        dbc.Col([
                            html.Div(['ЦЕЛЕВЫЕ КЛИЕНТЫ'], className='title-box color-red'),
                            html.Div([], className='border'),
                            html.Div(id='gen-target', className='number')
                        ], className='metr-stand', width={'size': 3})
                    ], justify='between'),

                    # селектор
                    dbc.Row([
                        html.Div(['ВЫБОР ПЕРИОДА']),
                        dbc.Col([time_selector
                        ], width={'size': 12})
                    ], className='selector-channel'),

                    # графики
                    dbc.Row([
                        dbc.Col([dcc.Graph(id='profit-chart')
                        ], className='chart-time', width={'size': 12})
                    ], className='margin-rows'),

                    dbc.Row([
                        dbc.Col(['Заработок с привлеченных клиентов/ Общие Затраты на привлечение'
                        ], className='chart-time', width={'size': 12})
                    ], className='margin-rows')

                ], className='main-col-right', width={'size': 5})

            ], className='stand-metr')



tab2_content = html.Div([

    dbc.Row([
        dbc.Col([dcc.Graph(id='chart-pred-fig6')
                 ], width={'size': 4}),

        dbc.Col([dcc.Graph(id='chart-pred-fig7')
                ], width={'size': 4}),

        dbc.Col([dcc.Graph(id='chart-pred-fig8')
                 ], width={'size': 4}),
    ]),
    dbc.Row([


        dbc.Col([dcc.Graph(id='chart-pred-fig9')
                 ], width={'size': 4}),

        dbc.Col([dcc.Graph(id='chart-pred-fig10')
                 ], width={'size': 4}),

        dbc.Col([dcc.Graph(id='chart-pred-fig11')
                 ], width={'size': 4})
    ]),

    dbc.Row([
        dbc.Col([dcc.Graph(id='chart-pred-fig12')
                 ], width={'size': 4}),
        dbc.Col([
                dbc.Button('Apply', id='submit-val', n_clicks=0,
                           color="secondary")], width={'size': 4})
    ])
])


"""  LAYOUT  """

app.layout = html.Div([
    # header
    dbc.Row([], className='header'),
    dbc.Row([
        #barside
        dbc.Col([], className='barside', width={'size': 1}),

        #main_side
        dbc.Col([
                dbc.Tabs([
                        dbc.Tab(tab1_content, label='CHARTS'),
                        dbc.Tab(tab2_content, label='PREDICT')
                        # dbc.Tab(tab3_content, label='About')
                        ]),
        ])
    ], className='wrapper')
             ], className='main-wrapper')



'''   FIGURS   '''

def plot_predict_figurs(data, el):

    input_names = ["create_date"]
    output_names = ["targ_leads"]

    sort_data = data[data['type_channel'] == el]

    train_data = sort_data[input_names + output_names]

    train_data.rename(columns={'create_date': 'ds', 'targ_leads': 'y'}, inplace=True)


    with open(f'{el}_model.json', 'r') as fin:
      m = model_from_json(fin.read())

    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)

    forecast.rename(columns={'ds': 'Date', 'trend': 'Trend'}, inplace=True)

    fig = px.line(forecast.iloc[-7:,:],
                   x='Date',
                   y='Trend',
                   title = f'Тренд для канала "{el}"'
                   )

    return fig



'''  CALLBACKS  '''

@app.callback(
    [Output(component_id='romi', component_property='children'),
     Output(component_id='avg-bill', component_property='children'),
     Output(component_id='conv', component_property='children'),
     Output(component_id='gen-target', component_property='children'),
     Output(component_id='dist-temp-chart', component_property='figure'),
     Output(component_id='pie-chart', component_property='figure'),
     Output(component_id='profit-chart', component_property='figure')],
    [Input(component_id='channel-selector', component_property='value'),
     Input(component_id='time-slider', component_property='value')])

def update_dist_temp_chart(channel_selector, time_slider):
    # Фильтрация датасета


    chart_data = df[(df['create_date'] > pd.to_datetime(dt.date.fromtimestamp(time_slider[0]))) &
                    (df['create_date'] < pd.to_datetime(dt.date.fromtimestamp(time_slider[1]))) &
                    (df['type_channel'].isin(channel_selector))]


    time_data = chart_data.copy()
    time_data['create_date'] = time_data['create_date'].dt.strftime('%m-%Y')
    time_data = time_data.groupby(['create_date', 'type_channel'])['targ_leads'].sum()
    time_data = time_data.reset_index()
    time_data['create_date'] = time_data['create_date'].astype('datetime64[ns]')
    time_data = time_data.sort_values(by='create_date')

    profit_data = chart_data.copy()
    profit_data['mean_targ_vs_cost'] = profit_data['avg_bill'] * profit_data['targ_leads'] - profit_data['click_price'] * profit_data['conversion']
    profit_data = profit_data.groupby(['type_channel'])['mean_targ_vs_cost'].median()

    '''  METRICS  '''

    ROMI = (sum(chart_data['avg_bill'] * chart_data['targ_leads']) - sum(chart_data['click_price'] * chart_data['conversion'])) * 100 / sum(
        chart_data['click_price'] * chart_data['conversion'])
    gen_conv = sum(chart_data['conversion'])
    gen_target_lead = sum(chart_data['targ_leads'])

    html1 = [html.Div([f'{round(ROMI,1)}%'])]
    html2 = [html.Div([f'{int(round(np.mean(chart_data["avg_bill"], 0)))}'])]
    html3 = [html.Div([f'{gen_conv}'])]
    html4 = [html.Div([f'{gen_target_lead}'])]




    ''' FIGURS '''

    fig1 = px.line(time_data,
                   x='create_date',
                   y='targ_leads',
                   color='type_channel'
                   )
    fig1.update_layout(
                       margin=dict(l=0, r=20, t=20, b=20),
                       title_font_size=14,
                       legend_font_size=10
    )


    fig2 = px.pie(chart_data, values='targ_leads', names='type_channel',
                  title='Распределение трафика по каналам'
                  )
    fig2.update_layout(legend_orientation="v",
                       # legend=dict(x=1, xanchor="center"),
                       margin=dict(l=0, r=0, t=20, b=0),
                       # textinfo='percent',
                       title_font_size=14,
                       legend_font_size=10)
    # fig1.update_layout(template=CHARTS_TEMPLATE)

    fig3 = px.bar(profit_data,
                  y=profit_data,
                  x=profit_data.index,
                  color=profit_data.index,
                  title="МАРЖА")

    fig3.update_layout(legend_orientation="v",
                       # legend=dict(x=1, xanchor="center"),
                       margin=dict(l=0, r=0, t=40, b=0),
                       # textinfo='percent',
                       title_font_size=14,
                       legend_font_size=10)


    # html1 = [html.H4("Planet Temperature ~ Distance from the Star"),
    #          dcc.Graph(figure=fig1)]
    return html1, html2, html3, html4, fig1, fig2, fig3


@app.callback(
    [Output(component_id='chart-pred-fig6', component_property='figure'),
     Output(component_id='chart-pred-fig7', component_property='figure'),
     Output(component_id='chart-pred-fig8', component_property='figure'),
     Output(component_id='chart-pred-fig9', component_property='figure'),
     Output(component_id='chart-pred-fig10', component_property='figure'),
     Output(component_id='chart-pred-fig11', component_property='figure'),
     Output(component_id='chart-pred-fig12', component_property='figure')],
    [Input(component_id='submit-val', component_property='n_clicks')])

def update_prediction_figurs(channel_selector):
    # Фильтрация датасета


    # chart_data = df[(df['create_date'] > pd.to_datetime(dt.date.fromtimestamp(time_slider[0]))) &
    #                 (df['create_date'] < pd.to_datetime(dt.date.fromtimestamp(time_slider[1]))) &
    #                 (df['type_channel'].isin(channel_selector))]
    #
    #
    # time_data = chart_data.copy()
    # time_data['create_date'] = time_data['create_date'].dt.strftime('%m-%Y')
    # time_data = time_data.groupby(['create_date', 'type_channel'])['targ_leads'].sum()
    # time_data = time_data.reset_index()
    # time_data['create_date'] = time_data['create_date'].astype('datetime64[ns]')
    # time_data = time_data.sort_values(by='create_date')

    fig6 = plot_predict_figurs(df, 'контекстная реклама ')
    fig7 = plot_predict_figurs(df, 'VK, Telegram')
    fig8 = plot_predict_figurs(df, 'нативная реклама')
    fig9 = plot_predict_figurs(df, 'CPA')
    fig10 = plot_predict_figurs(df, 'медийная реклама')
    fig11 = plot_predict_figurs(df, 'геймификация')
    fig12 = plot_predict_figurs(df, 'Programmatic')

    return fig6, fig7, fig8, fig9, fig10, fig11, fig12


if __name__ == '__main__':
    app.run_server(debug=True)


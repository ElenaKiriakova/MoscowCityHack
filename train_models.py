import numpy as np
import pandas as pd
'''!pip install prophet'''
import plotly.express as px


import prophet
from prophet.serialize import model_to_json, model_from_json


"""  FUNCTIONS  """
def plot_predict_figurs(data, el):
    '''Обучение модели для предсказания линии тренда'''

    input_names = ["create_date"]
    output_names = ["targ_leads"]

    sort_data = data[data['type_channel'] == el]

    train_data = sort_data[input_names + output_names]

    train_data.rename(columns={'create_date': 'ds', 'targ_leads': 'y'}, inplace=True)

    m = Prophet(daily_seasonality=True)
    m.fit(train_data)

    with open(f'{el}_model.json', 'w') as fout:
      fout.write(model_to_json(m))

    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)

    forecast.rename(columns={'ds': 'Date', 'trend': 'Trend'}, inplace=True)

    fig = px.line(forecast.iloc[-7:,:],
                   x='Date',
                   y='Trend',
                   title = f'Тренд для канала "{el}"'
                   )

    return fig
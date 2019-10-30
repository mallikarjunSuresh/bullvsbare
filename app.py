# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import requests
from bs4 import BeautifulSoup
import pandas_datareader as pdr
from financialreportingdfformatted import save_sp500_stocks_info
from datetime import datetime, timedelta

from math import log
from math import sqrt
from math import exp
from scipy.stats import norm
import mibian

n = norm.pdf
N = norm.cdf
click_val = ''

def bs_price(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
    d2 = d1-v*sqrt(T)
    if cp_flag == 'c':
        price = S*exp(-q*T)*N(d1)-K*exp(-r*T)*N(d2)
    else:
        price = K*exp(-r*T)*N(-d2)-S*exp(-q*T)*N(-d1)
    return price

def bs_vega(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (log(S/K)+(r+v*v/2.)*T)/(v*sqrt(T))
    return S * sqrt(T)*n(d1)

app = dash.Dash(__name__)
server = app.server
app.config['suppress_callback_exceptions']=True

model = ''
df = pdr.get_data_yahoo('ibm',start=datetime(2006, 10, 1), end=datetime.now())
df.reset_index(inplace=True)
# df2 = pd.read_csv("dataset_Facebook.csv",";")
#
# df_ml = df2.copy()
#
# lb_make = LabelEncoder()
# type = df_ml['Type']
# df_ml["Type"] = lb_make.fit_transform(type)
# df_ml = df_ml.fillna(0)
#
# X = df_ml.drop(['like'], axis = 1).values
# Y = df_ml['like'].values
#
# X = StandardScaler().fit_transform(X)
#
# X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
#
# randomforest = RandomForestRegressor(n_estimators=500,min_samples_split=10)
# randomforest.fit(X_Train,Y_Train)
#
# p_train = randomforest.predict(X_Train)
# p_test = randomforest.predict(X_Test)
#
# train_acc = r2_score(Y_Train, p_train)
# test_acc = r2_score(Y_Test, p_test)

app.layout = html.Div([html.H1("Stock Data Analysis", style={"textAlign": "center"}), dcc.Markdown('''
Welcome to interactive dashboard for stock anlayis. This dashboard is divided in 3 main tabs. In the first tab we are proving graphical information of the selected stock.Which is listed dynamically form wikipedia using webscrapping.
Using the second tab, you can do prediciton of stock price for the selected stock. Finally, in the third tab forcasting derivative price is done. 
All the data displayed in this dashboard is fetched, processed and updated using Python (eg. ML models are trained in real time!).
''')  ,
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Stock Prices', children=[
html.Div([html.H1("Dataset Introduction", style={'textAlign': 'center'}),
    html.H1("Stocks High vs Lows", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown',options=save_sp500_stocks_info(),
        multi=True,value=['ibm'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='highlow'),
    html.H1("Stock Market Volume", style={'textAlign': 'center', 'padding-top': 5}),
    dcc.Dropdown(id='my-dropdown2',options=save_sp500_stocks_info(),
        multi=True,value=['ibm'],style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "80%"}),
    dcc.Graph(id='volume'),
    html.H1("Scatter Analysis", style={'textAlign': 'center', 'padding-top': -10}),
    dcc.Dropdown(id='my-dropdown3',
                 options= save_sp500_stocks_info(),
                 value= 'ibm',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
    dcc.Dropdown(id='my-dropdown4',
                 options= save_sp500_stocks_info(),
                 value= 'ibm',
                 style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "45%"}),
  dcc.RadioItems(id="radiob", value= "High", labelStyle={'display': 'inline-block', 'padding': 10},
                 options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"} , {'label': "Volume", 'value': "Volume"}],
 style={'textAlign': "center", }),
    dcc.Graph(id='scatter')
], className="container"),
]),
# dcc.Tab(label='Performance Metrics', children=[
# html.Div([html.H1("Facebook Metrics Distributions", style={"textAlign": "center"}),
#             html.Div([html.Div([dcc.Dropdown(id='feature-selected1',
#                                              options=[{'label': i.title(), 'value': i} for i in
#                                                       df2.columns.values[1:]],
#                                              value="Type")],
#                                style={"display": "block", "margin-left": "auto", "margin-right": "auto",
#                                       "width": "80%"}),
#                       ],),
#             dcc.Graph(id='my-graph2'),
# dash_table.DataTable(
#     id='table3',
#     columns=[{"name": i, "id": i} for i in df.describe().reset_index().columns],
#     data= df.describe().reset_index().to_dict("rows"),
# ),
#             html.H1("Paid vs Free Posts by Category", style={'textAlign': "center", 'padding-top': 5}),
#      html.Div([
#          dcc.RadioItems(id="select-survival", value=str(1), labelStyle={'display': 'inline-block', 'padding': 10},
#                         options=[{'label': "Paid", 'value': str(1)}, {'label': "Free", 'value': str(0)}], )],
#          style={'textAlign': "center", }),
#      html.Div([html.Div([dcc.Graph(id="hist-graph", clear_on_unhover=True, )]), ]),
#         ], className="container"),
# ]),
dcc.Tab(label='Machine Learning', children=[
html.Div([html.H1("Machine Learning", style={"textAlign": "center"}), html.H2("ARIMA Time Series Prediction", style={"textAlign": "left"}),
    dcc.Dropdown(id='my-dropdowntest',options= save_sp500_stocks_info(),value = 'ibm',
                 # [{'label': 'Tesla', 'value': 'TSLA'},{'label': 'Apple', 'value': 'AAPL'},{'label': 'Facebook', 'value': 'FB'},{'label': 'Microsoft', 'value': 'MSFT'}],
                style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "50%"}),
          dcc.RadioItems(id="radiopred", value="High", labelStyle={'display': 'inline-block', 'padding': 10},
                         options=[{'label': "High", 'value': "High"}, {'label': "Low", 'value': "Low"},
                                  {'label': "Volume", 'value': "Volume"}], style={'textAlign': "center", }),
    dcc.Graph(id='traintest'), dcc.Graph(id='preds'),
html.H2("Performance Metrics Regression Prediction", style={"textAlign": "left"}), html.P("In this model we are using ARIMA time series analysis for the performance of stock"),
    html.P("In order to achieve these results, all the not a numbers (NaNs) have been eliminated, categorical data has been encoded and the data has been normalized. The R2 score has been used as metric for this exercise and a Train/Test split ratio of 70:30% was used.")
], className="container")
]),

dcc.Tab(label='Derivatives', children=[
dcc.Tabs(id='Derivatives', children=[
dcc.Tab(label='Calculate Volatility', children=[
html.Div([html.H1('Volatility Calculation'),


          dcc.Input(id='target-value', value=17.5, type='number', placeholder='option-price'),html.Div(id='div-target-value'),
          dcc.Input(id='call-put', value='c', type='text', placeholder='call-put'), html.Div(id='div-call-put'),
          dcc.Input(id='underlying-price', value=356, type='number', placeholder='underlying-price'),html.Div(id='div-underlying-price'),
          dcc.Input(id='strike-price', value=350, type='number', placeholder='strike-price'),html.Div(id='div-strike-price'),
          dcc.Input(id='time-to-mature', value=50, type='number', placeholder='time-to-mature-in-days'),html.Div(id='div-time-to-mature'),
          dcc.Input(id='interest-rate', value=0.002, type='number', placeholder='interest-rate'),html.Div(id='div-interest-rate'),
          html.Button(id='submit-button', n_clicks=0, children='Submit')

],
         className="container")
]),

dcc.Tab(label='Calculate Option Price', children=[
html.Div([html.H1('Option Price Calculation'),


          dcc.Input(id='volatility', value=27.345982690486065, type='number', placeholder='Volatility in percentage'),html.Div(id='div_volatility'),
          dcc.Input(id='call_put', value='c', type='text', placeholder='call-put'), html.Div(id='div_call_put'),
          dcc.Input(id='underlying_price', value=356, type='number', placeholder='underlying-price'),html.Div(id='div_underlying_price'),
          dcc.Input(id='strike_price', value=350, type='number', placeholder='strike-price'),html.Div(id='div_strike_price'),
          dcc.Input(id='time_to_mature', value=50, type='number', placeholder='time-to-mature-in-days'),html.Div(id='div_time_to_mature'),
          dcc.Input(id='interest_rate', value=0.002, type='number', placeholder='interest-rate'),html.Div(id='div_interest_rate'),
          html.Button(id='submit_button', n_clicks=0, children='Submit')

],
         className="container")
])

])
])



])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
#    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        df = pdr.get_data_yahoo(str(stock), start=datetime(2006, 10, 1), end=datetime.now())
        df.reset_index(inplace=True)
        trace1.append(go.Scatter(x=df["Date"],y=df["High"],mode='lines',
            opacity=0.7,name=f'High {stock}',textposition='bottom center'))
        trace2.append(go.Scatter(x=df["Date"],y=df["Low"],mode='lines',
            opacity=0.6,name=f'Low {stock}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"High and Low Prices for {', '.join(str(stock) for stock in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    # dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        df = pdr.get_data_yahoo(str(stock), start=datetime(2006, 10, 1), end=datetime.now())
        df.reset_index(inplace=True)
        trace1.append(go.Scatter(x=df["Date"],y=df["Volume"],mode='lines',
            opacity=0.7,name=f'Volume {stock}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,title=f"Market Volume for {', '.join(str(stock) for stock in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Transactions Volume"} ,   paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('scatter', 'figure'),
              [Input('my-dropdown3', 'value'), Input('my-dropdown4', 'value'), Input("radiob", "value"),])
def update_graph(stock, stock2, radioval):
    # dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    if (stock == None) or (stock2 == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        df = pdr.get_data_yahoo(str(stock), start=datetime(2006, 10, 1), end=datetime.now())
        df.reset_index(inplace=True)
        df2 = pdr.get_data_yahoo(str(stock2), start=datetime(2006, 10, 1), end=datetime.now())
        df2.reset_index(inplace=True)
        trace1.append(go.Scatter(x=df[radioval][-1000:], y=df2[radioval][-1000:],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} of {stock} vs {stock2} Over Time (1000 iterations)",
                xaxis={"title": stock,}, yaxis={"title": stock2},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure

# @app.callback(
#     dash.dependencies.Output('my-graph2', 'figure'),
#     [dash.dependencies.Input('feature-selected1', 'value')])
# def update_graph(selected_feature1):
#     if selected_feature1 == None:
#         selected_feature1 = 'Type'
#         trace = go.Histogram(x= df2.Type,
#                              marker=dict(color='rgb(0, 0, 100)'))
#     else:
#         trace = go.Histogram(x=df2[selected_feature1],
#                          marker=dict(color='rgb(0, 0, 100)'))
#     return {
#         'data': [trace],
#         'layout': go.Layout(title=f'Metric: {selected_feature1.title()}',
#                             colorway=["#EF963B", "#EF533B"], hovermode="closest",
#                             xaxis={'title': "Distribution", 'titlefont': {'color': 'black', 'size': 14},
#                                    'tickfont': {'size': 14, 'color': 'black'}},
#                             yaxis={'title': "Frequency", 'titlefont': {'color': 'black', 'size': 14, },
#                                    'tickfont': {'color': 'black'}},     paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)')}


# @app.callback(
#     dash.dependencies.Output("hist-graph", "figure"),
#     [dash.dependencies.Input("select-survival", "value"),])
# def update_graph(selected):
#     dff = df2[df2["Paid"] == int(selected)]
#     trace = go.Histogram(x=dff["Type"], marker=dict(color='rgb(0, 0, 100)'))
#     layout = go.Layout(xaxis={"title": "Post distribution categories", "showgrid": False},
#                        yaxis={"title": "Frequency", "showgrid": False},    paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)' )
#     figure2 = {"data": [trace], "layout": layout}
#
#     return figure2


@app.callback(Output('traintest', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock , radioval):
    # {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    radio = {"High": "High  Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    trace1 = []
    trace2 = []
    df = pdr.get_data_yahoo(str(stock),start=datetime(2006, 10, 1),end=datetime.now())
    df.reset_index(inplace=True)
    count = df['Date'].count() + 1
    train_data = df[:][0:int(count * 0.8)]
    test_data = df[:][int(count * 0.8):]
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        trace1.append(go.Scatter(x=train_data['Date'],y=train_data[radioval], mode='lines',
            opacity=0.7,name=f'Training Set',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y=test_data[radioval],mode='lines',
            opacity=0.6,name=f'Test Set',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} Train-Test Sets for {stock}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


@app.callback(Output('preds', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value"),])
def update_graph(stock, radioval):
    radio = {"High": "High Prices", "Low": "Low Prices", "Volume": "Market Volume", }
    #{"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    if (stock == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:
        df = pdr.get_data_yahoo(str(stock), start=datetime(2006, 10, 1), end=datetime.now())
        df.reset_index(inplace=True)
        count = df['Date'].count() + 1
        train_data = df[:][0:int(count * 0.8)]
        test_data = df[:][int(count * 0.8):]
        train_ar = train_data[radioval].values
        test_ar = test_data[radioval].values
        history = [x for x in train_ar]
        predictions = list()
        for t in range(len(test_ar)):
            global model
            model = ARIMA(history, order=(3, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_ar[t]
            history.append(obs)
        error = mean_squared_error(test_ar, predictions)
        trace1.append(go.Scatter(x=test_data['Date'],y=test_data['High'],mode='lines',
            opacity=0.6,name=f'Actual Series',textposition='bottom center'))
        trace2.append(go.Scatter(x=test_data['Date'],y= np.concatenate(predictions).ravel(), mode='lines',
            opacity=0.7,name=f'Predicted Series (MSE: {error})',textposition='bottom center'))
        traces = [trace1, trace2]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} ARIMA Predictions vs Actual for {stock}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":"Price (USD)"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure
@app.callback(
            Output(component_id='div-target-value', component_property='children'),
            [Input(component_id='target-value', component_property='value'),
             Input(component_id='call-put', component_property='value'),
             Input(component_id='underlying-price', component_property='value'),
             Input(component_id='strike-price', component_property='value'),
             Input(component_id='time-to-mature', component_property='value'),
             Input(component_id='interest-rate', component_property='value'),
             Input('submit-button', 'n_clicks')]
)

def find_vol(target_value, call_put, S, K, T, r, clicked):

    if clicked and target_value and call_put and S and K and T and T >= 0 and r:

        MAX_ITERATIONS = 100
        PRECISION = 1.0e-5
        sigma = 0.5
        target_value = float(target_value)
        S = float(S)
        K = float(K)
        T = float(T)
        T = T/365
        r = float(r)
        for i in range(0, MAX_ITERATIONS):
            price = bs_price(call_put, S, K, T, r, sigma)
            vega = bs_vega(call_put, S, K, T, r, sigma)

            price = price
            diff = target_value - price  # our root

            print(i, sigma, diff)

            if (abs(diff) < PRECISION):
                sigma = sigma*100
                return 'Volatility is "{}"'.format(sigma)
            sigma = sigma + diff / vega  # f(x) / f'(x)
        sigma = sigma * 100
        return 'Volatility is "{}"'.format(sigma)

    else:
        return ''

@app.callback(
    Output(component_id='div_volatility', component_property='children'),
    [Input(component_id='volatility', component_property='value'),
    Input(component_id='call_put', component_property='value'),
    Input(component_id='underlying_price', component_property='value'),
    Input(component_id='strike_price', component_property='value'),
    Input(component_id='time_to_mature', component_property='value'),
    Input(component_id='interest_rate', component_property='value'),
    Input('submit_button', 'n_clicks')]
)
def find_price(target_value, call_put, S, K, T, r, clicked):

        if clicked and target_value and call_put and S and K and T and T >= 0 and r:

            sigma = mibian.BS([S, K, r, T], volatility=target_value)
            if call_put == 'c':
                return sigma.callPrice
            if call_put == 'p':
                return sigma.putPrice

        else:
            return ''

    # value wasn't found, return best guess so far
 #   return 'Volatility is "{}"'.format(sigma)

if __name__ == '__main__':
    app.run_server(debug=True)
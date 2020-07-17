####  COVID19 2.0 ####

## official doc
## https://plotly.com/python/reference/
## https://dash.plotly.com/dash-html-components

## WORKING ON Heroku app!
## DEPLOY PROBLEM (SEEMS THAT matplotlib IS NOT SUPPORTED IN HEROKU)
## CHANGE COLOR SCALE TO PLOTLY (DON'T IMPORT matplotlib)
## https://plotly.com/python/builtin-colorscales/
## https://plotly.com/python/discrete-color/

## HOVER AND HIGHLIGHT
# https://stackoverflow.com/questions/53327572/how-do-i-highlight-an-entire-trace-upon-hover-in-plotly-for-python
# https://plotly.com/javascript/plotlyjs-events/

# https://github.com/COVID19Tracking/covid-tracking-dash/blob/master/covid_tracking/app.py
# http://35.212.27.3:8050/
## COOL WAY TO DO THE SIMILAR THING. CAN LEARN MARKDOWN FROM THIS


# import requests
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_html_components as html
from dash.dash import no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly_express as px
import dash_table

# import matplotlib as mpl
# import matplotlib.pyplot as plt
import math

## Load the data
data_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
data_set = pd.read_csv(data_url, dtype={'fips': 'category', 'state': 'category'})
df = data_set.copy()


#### ETL ####
### Heat map
def ETL_df(df):
    ## date time convert
    df.index = pd.to_datetime(df['date'])
    df = df.drop('date', axis='columns')
    df = df.sort_values(by=['date', 'fips'])

    ## filter only the states
    import us
    state_dic = {s.fips: s.name for s in us.states.STATES}
    df = df[df['fips'].isin(set(state_dic.keys()))]  # only has 50 states data
    df['fips'].cat.remove_unused_categories(inplace=True)
    df['state'].cat.remove_unused_categories(inplace=True)

    # Complete 50 states records start from 2020-03-17
    df = df.loc['2020-03-17':]
    df_case = df.pivot_table(values='cases', index=['date'], columns='state').rolling(window=7).mean().diff().iloc[7:]

    # replace some negative value with 0
    df_case = df_case.applymap(lambda x: 0 if x < 0 else x)

    return df_case


def state_pop_dict():
    df_pop = pd.read_excel(
        'https://www2.census.gov/programs-surveys/popest/tables/2010-2019/state/totals/nst-est2019-01.xlsx', header=3)
    df_pop = df_pop[[2019, 'Unnamed: 0']].iloc[:-7].set_index('Unnamed: 0')
    df_pop.index = df_pop.index.str.replace('.', '')
    df_pop = df_pop.iloc[5:].drop('District of Columbia', axis='index')
    ## make dictionary
    state_pop_dict = df_pop.to_dict('dict')[2019]
    return state_pop_dict


def data_process_avg(complete_df, pop_dict):
    for c in list(complete_df.columns):
        try:
            complete_df[c] = round(complete_df[c] / pop_dict.get(c) * 1000000, 3)
        except:
            complete_df[c + '_no_pop_value'] = complete_df[c]
    return complete_df


def case_data_process(df_case):
    ## cases data
    case_array = np.array([df_case[col] for col in df_case_pop])

    ## scale with all states (find the smallest daily increase and biggest daily increase; use this as scale)
    # max: 590.443, min: 0
    def standardized_all(array):
        return (array - np.min(array, axis=0)) / np.ptp(array, axis=None)

    case_array_std = standardized_all(case_array)  # max: 590.443, min: 0

    ## date
    date_list = list(dict.fromkeys(df_case.index))  # check!

    ## state
    state_list = list(df_case.columns)
    return case_array, case_array_std, date_list, state_list


### Real map
def df_map():
    df_map = pd.DataFrame(df_case_pop.iloc[-1])
    df_map.columns = ['case']
    df_map = df_map.reset_index()
    return df_map


def get_phase_dict():
    from bs4 import BeautifulSoup
    import requests
    url = "https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html"
    res = requests.get(url)
    #     print(res)
    # the number 200 is http status code

    soup = BeautifulSoup(res.text, 'lxml')
    sr = soup.select(".g-name , .g-cat-subhed span")
    scrap_list = [i.text for i in sr]
    scrap_list = [x for x in scrap_list if x not in ['District of Columbia', 'Puerto Rico']]

    import us
    scrap_set = set(scrap_list)
    state_set = set([s.name for s in us.states.STATES])
    phase_set = scrap_set - state_set

    phase_dict = {}
    for i in scrap_list:
        if i in phase_set:
            phase = i
            continue
        phase_dict[i] = phase
    # print('complete extraction')
    return phase_dict

def get_phase_link_df():
    from bs4 import BeautifulSoup
    import requests
    import re

    url = "https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'lxml')

    # state
    sr1 = soup.select('.g-name')
    scrap_list1 = [i.text for i in sr1]

    # link
    sr2 = soup.select('.g-link a', attrs={'href': re.compile("^http://")})
    scrap_list2 = [link.get('href') for link in sr2]
    link_list = [f'''[News Link]('{l}')''' for l in scrap_list2]
    df_link = pd.DataFrame(np.array([scrap_list1, link_list]).T, index=scrap_list1, columns=['state', 'link'])
    df_link.drop(['District of Columbia', 'Puerto Rico'], inplace=True)
    return df_link


def df_map_ETL(df_map):
    ## death data with most recent records
    death_dict = df[df['date'] == df['date'].max()][['state', 'deaths']].set_index('state').T.to_dict('index')['deaths']
    death_dict = {s: round(death_dict.get(s) / state_pop_dict.get(s) * 1000000, 3) for s in df_map['state']}
    ## abbre data for plotting function
    import us
    abbr_dict = us.states.mapping('name', 'abbr')

    ## pipeline
    df_map['abbr'] = df_map['state'].map(abbr_dict)
    df_map['phase'] = df_map['state'].map(phase_dict)
    df_map['death'] = round(df_map['state'].map(death_dict), 0).astype(int)
    df_map['case'] = round(df_map['case']).astype(int)
    # for hoverinfo
    df_map['text'] = (
                '<I><b>' + df_map['state'].astype('str') + '</b></I><br>' + 'New Cases per 1M Resident: '
                + df_map['case'].astype('str')
                + '<br>' + 'New Death per 1M Resident: ' + df_map['death'].astype('str') + '<br>' + 'Phase: ' + df_map[
                    'phase'] + '<extra></extra>')
    return df_map

### individual line
def standardized_row(array):
    return ((array.T - np.min(array.T, axis=0))/np.ptp(array, axis=1)).T

### Final dataset
## heat map
df_case = ETL_df(df)
state_pop_dict = state_pop_dict()
df_case_pop = data_process_avg(df_case, state_pop_dict)
case_array, case_array_std, date_list, state_list = case_data_process(df_case_pop)

## real map
phase_dict = get_phase_dict()
df_map = df_map_ETL(df_map())

## individual line
case_array_std_row = standardized_row(case_array)
df_case_std_row = pd.DataFrame(case_array_std_row, index=state_list)

## other info
current_date = date_list[-1].strftime('%Y/%m/%d')
df_link = get_phase_link_df()

####
def case_data_process_tt(df_case_pop):
    ## rank by higest cases date
    sort_state_list = df_case_pop.idxmax().sort_values().index

    ## cases data
    case_array = np.array([df_case_pop[state] for state in sort_state_list])

    ## scale with all states (find the smallest daily increase and biggest daily increase; use this as scale)
    # max: 590.443, min: 0
    def standardized_all(array):
        return (array - np.min(array, axis=0)) / np.ptp(array, axis=None)

    case_array_std = standardized_all(case_array)  # max: 590.443, min: 0

    ## date
    date_list = list(dict.fromkeys(df_case_pop.index))  # check!

    ## state
    state_list = sort_state_list
    return case_array, case_array_std, date_list, state_list


#### Alarm function
# find out which states break their state records in the last 5 days (meaning highly dangerous states!)
def alarm_state_list():
    dict_alarm = (df_case_pop.idxmax().sort_values() > df_case_pop.tail(5).index[0]).to_dict()
    alarm_state_list = [k for k, v in dict_alarm.items() if v == True]
    return alarm_state_list
alarm_state_list = alarm_state_list()



###############

####  drop down list function
# create the option list

state_options = [dict(label=s, value=s) for s in state_list]
dropdown_function_state = dcc.Dropdown(id='state_selection',
                                       options=state_options,
                                       multi=True,
                                       value=['New York', 'Arizona'],
                                       placeholder="Pick states",
                                       # className="dcc_control"
                                       )

### test ###


####  Initiate the app
#### style setting
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app = dash.Dash(external_stylesheets=[dbc.themes.GRID, dbc.themes.BOOTSTRAP])
server = app.server
app.title = 'states spot the curve'

# in fact, html.Div() don't need children in my test
# original version
# app.layout = html.Div([
#     html.H1('TEST: This is a title'),
#     html.Div('This is some word'),
#     html.P(['Please select a country' + ':', dropdown_function]),
#     html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
#     dcc.Graph(id='output-graph', animate=None),
# ])

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div(
                    'Spot the Curves in US',
                    style={'font-size': '30px', 'font-style': 'bold'}
                ),
                dcc.Markdown(
                    '''
                    A Better Overview of COVID-19 in Each State 
                    *Made by [Jeff Lu](https://www.linkedin.com/in/jefflu-chia-ching-lu/)*
                    ''',
                    style={'font-style': 'italic', 'display': 'inline'}),
            ]),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H4('Purpose and Method', style={'font-style': 'bold'}),
                dcc.Markdown(
                    '''
                To better understand COVID-19 in the US, this dashboard presents overview of state-level data 
                with complementary information. This dashboard aims to show individual trend within each state, 
                provide timely state-level phase information and **relieve people from overwhelming news from media 
                outlets**. \n 
                
                The major metrics include:
                - ```7-Day Moving Average```: A series of averages of daily increase in cases across the time 
                - ```Phase```: The phase of the Coronavirus restrictions issued by each state government (defined and
                 produced by New York Times; The bottom has a list of latest news about each state)
                '''
                ),
                html.H4('Reference and Relevant Reading', style={'font-style': 'bold', "margin-top": "30px"}),
                html.Div(dcc.Markdown(
                    '''
                    The state-level data is from [New York Times GitHub](https://github.com/nytimes/covid-19-data). 
                    The information about phase of each state is 
                    scrapped from [New York Times](https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html).
                    Here is another [country-level dashboard](https://spot-the-curve-coronavirus.herokuapp.com/) and 
                    the [Medium article](https://towardsdatascience.com/spot-the-curve-visualization-of-cases-data-on-coronavirus-8ec7cc1968d1?source=friends_link&sk=4f984ca1c1e4df9535b33d9ccab738ee) about interpreting Coronavirus data. 
                    
                '''
                ),
                    style={'font-size': '13px', 'color': 'grey'}
                ),
            ], width=5, style={'paddingRight': '50px'}
            ),
            dbc.Col([
                html.H4(
                    'Pick the States and Compare the Curves!',
                    style={'font-style': 'bold', "margin-top": "0px"}),
                html.Div(dcc.Markdown(
                    '''
                    The distribution of daily growth is normalized within the state (__displayed by percentage__) 
                '''
                ),
                    style={'font-size': '13px', 'color': 'grey'}
                ),
                    dbc.Row([dbc.Col(html.Div(dropdown_function_state)),
                             dbc.Col(dbc.Button('Submit', id='button', style={}), width=2)]),
                dcc.Graph(id='output-graph3', animate=None)
            ], width=7, style={'paddingLeft': '50px'}), ],
        ),
        dbc.Row([
            dbc.Col(html.H4('Daily New Cases per 1M Resident', style={'font-style': 'bold'}), width=5
                ),
            dbc.Col([
                html.Div(
                    dbc.Button('Alphabetical', id='button_alpha', outline=True, color="primary", style={}),
                ),
            ],  width={"size": 1, "offset": 5}
            ),
            dbc.Col([
                html.Div(
                    dbc.Button('Rank', id='button_order', outline=True, color="primary", style={}),
                    ),
            ], width={"size": 1, "offset": 0}
            ),
        ], style={'paddingTop': '50px'}),
        dbc.Row([
            dbc.Col([
                html.Div(
                    dcc.Graph(id='output-graph1', animate=None)
                ),
            ]),
        ], ),
        dbc.Row([
            dbc.Col(html.H4('Latest Information Map', style={'font-style': 'bold'}), width=5
                    ),
            dbc.Col([
                html.Div(
                    dbc.Button('Daily', id='button_info', outline=True, color="primary", style={'margin-left': '50px'}),
                ),
            ], width={"size": 1, "offset": 5}
            ),
            dbc.Col([
                html.Div(
                    dbc.Button('Phase', id='button_phase', outline=True, color="primary", style={'margin-left': '0px'}),
                ),
               ], width={"size": 1, "offset": 0}),
        ], style={'paddingTop': '70px'}),
        dbc.Row([
            dbc.Col([
                html.Div(
                    dcc.Graph(id='output-graph2', animate=None)
                ),
            ]),
        ], ),
        html.Div(dash_table.DataTable(id='datatable',
                             export_format="csv",
                             data=df_link.to_dict('records'),
                            # columns=[{"name": i, "id": i} for i in df_link.columns],
                             columns=[{"id": "state", "name": ["State"], "presentation": "markdown",},
                                      {"id": "link", "name": ["News Link"], "presentation": "markdown",}],
                                page_action='none',
                                style_table={'height': '300px', 'overflowY': 'auto'}
                        )),
    ], fluid=True,
        style={'paddingLeft': '200px', 'paddingRight': '200px', 'paddingBottom': '100px', 'paddingTop': '50px'})
])


## If using dash code to plot, then use html.Div
## If using plotly code to plot, then use dcc.Graph
## p.s. they have different components so watch out (children vs figure)

#### Plotting function ####

def create_case_heatmap(case_array, case_array_std, date_list, state_list):
    fig = go.Figure(data=go.Heatmap(
        z=case_array_std,
        x=date_list,
        y=state_list,
        colorscale='OrRd',
        hovertemplate='%{y}<br>'
                      + '%{x}<br>'
                      + 'New Cases: <b>%{text:.0f}'
                      + '<extra></extra>',
        text=np.array(list(reversed(case_array))),
        showscale=False,
    ))

    fig.update_layout(
        title={'text': '',
               # COVID - 19 New Cases per 1M Resident per Day
               # "yref": "paper",
               'y': 1,
               'x': 0.01,
               },
        xaxis_nticks=5,
        yaxis_categoryarray=list(reversed(state_list)),
        autosize=True,
        # width=1200,
        height=1000,
        margin={
            'autoexpand': True,
            'l': 5,
            'r': 2,
            't': 30,
            'b': 5,
        },
        # autosize=False,
        # height=350,
        # paper_bgcolor='#F7FBFE',  # canvas color
        # plot_bgcolor='#F7FBFE',  # plot color #D8D991 #F6EEDF #FFF8DE
    )

    annotations = []
    date_tt = df_case_pop.idxmax().sort_values()
    start_date = date_list[0]
    current_date = date_list[-1]

    #  emoji template: https://fsymbols.com/signs/stars/
    for i, c in enumerate(state_list):
        annotations.append(dict(xref='paper',
                                x=1 - (current_date - date_tt[c]) / (current_date - start_date),
                                y=c,
                                xanchor='right', yanchor='middle',
                                text='✦',
                                font=dict(family='Arial', color='gray',
                                          size=16), showarrow=False))
    annotations.append(dict(xref='paper', yref='paper',
                            x=1,
                            y=1.007, #paper 1
                            xanchor='right', yanchor='middle',
                            text='✦: The date having the highest daily increase within the state ',
                            font=dict(family='Arial', color='gray',
                                      size=15), showarrow=False))
    fig.update_layout(annotations=annotations)
    annotations = []

    return fig


def create_real_map(phase=False):
    if phase is False:
        fig = go.Figure(data=go.Choropleth(
            locations=df_map['abbr'],
            z=df_map['case'],
            text=df_map['text'],  # hover info
            locationmode='USA-states',
            colorscale='Reds',
            autocolorscale=False,
            marker_line_color='white',  # line markers between states
            colorbar_title="Daily Cases per 1M",
            hovertemplate='%{text}' + '<extra></extra>',
        ))
    else:
        color_list = [px.colors.qualitative.Vivid[i] for i in [3, 2, 10, 6]]
        phase_list = ['Reopened', 'Reopening', 'Pausing', 'Reversing']
        color_dict = {p: c for c, p in zip(color_list, phase_list)}

        fig = px.choropleth(locations=df_map['abbr'],
                            color=df_map['phase'],
                            scope="usa", locationmode="USA-states",
                            color_discrete_map=color_dict)
        fig.update_traces(marker_line_width=0.2, marker_opacity=0.9)

    fig.update_layout(
        title={'text': '', # f'Latest Daily Information({current_date})'
               # "yref": "paper",
               'y': 0.98,
               'x': 0.01,
               },
        geo=dict(
            scope='usa',
            projection=go.layout.geo.Projection(type='albers usa'),
            showlakes=True,  # lakes
            lakecolor='rgb(255, 255, 255)'),
        # paper_bgcolor='#F7FBFE',  # canvas color
        # plot_bgcolor='#F7FBFE',  # plot color #D8D991 #F6EEDF #FFF8DE
        # hoverlabel={'namelength': -1},
        # autosize=False,
        height=700,
        # autosize = True,
        margin={
            # 'autoexpand': True,
            'l': 0,
            'r': 0,
            't': 10, #20
            'b': 10,
        },
        # colorbar={'x':1},
        # coloraxis_colorbar={'thickness': 5, 'x':-1, 'xanchor':'right'},
    )
    return fig

### under development
def create_trend_line(df, selected_state_list):
    color_n = len(selected_state_list)
    cm = px.colors.qualitative.T10
    color_m = cm[0:color_n]

    fig = go.Figure()
    fig.layout.template = 'ggplot2'
    for i, c in enumerate(selected_state_list):
        fig.add_trace(go.Scatter(x=date_list, y=df.loc[c].values*100,
                                 text=[d.strftime("%Y-%m-%d") for d in date_list],
                                 mode='lines',
                                 # line_shape='spline',
                                 name=c,
                                 line={'color': color_m[i]},
                                 # hoverinfo="y+name",
                                 hovertemplate='%{text}: <b>%{y:.0f}%</b><extra></extra>', ))
        # '{:.0f}%'.format(df.loc[c]*100)
        # '%{x}: <b>%{y:.3f}</b>)<extra></extra>'
        fig.add_trace(go.Scatter(
            x=[date_list[-1]],
            y=[df.loc[c].values[-1]*100],
            name=c + ' endpoint',
            mode='markers',
            marker={'color': color_m[i], 'size': 10},
            hovertemplate='%{x}: <b>%{y:.0f}%</b><extra></extra>',
            showlegend=False,

        ))

    fig.update_layout(
        title={
            'text': '',
            'font': {'size': 20, 'family': 'Arial, sans-serif'},
            # "yref": "paper",
            'y': 0.98,
            'x':0.01,
            # 'xanchor': 'left',
            # 'yanchor': 'top',
            },
        xaxis={
            'title_text': '',
            # 'tickformat': '%b',
            # 'tickformatstops1': [dict(dtickrange=["M1", "M12"], value="%b %Y"),],

            'tickmode': 'auto',
            'ticks': 'outside',
            'tickcolor': '#F7FBFE',
            'title_standoff': 15,
            'nticks': 5,
            'showline': True,
            'showgrid': False,
            'showticklabels': True,
            'linecolor': 'rgb(204, 204, 204)',
            'linewidth': 2,
            'anchor': 'free',
            'position': 0.02,
        },
        yaxis={
            'title_text': '',
            'title_standoff': 5,
            'ticks': 'outside',
            'tickcolor': '#F7FBFE',
            'gridcolor': '#EEEEEE',
            'showgrid': False,
            'zeroline': False,
            'showline': False,
            'showticklabels': False, },
        showlegend=True,
        # legend_borderwidth=5,
        paper_bgcolor='#FFFFFF',  # canvas color #F7FBFE
        plot_bgcolor='#FFFFFF',  # plot color #D8D991 #F6EEDF #FFF8DE #F7FBFE
        hoverlabel={'namelength': -1},
        # autosize=False,
        # height=350,
        margin={
            'autoexpand': True,
            'l': 0,
            'r': 20,
            't': 10,
            'b': 0,
                }
    )

    # Adding labels
    # annotations = []
    # for i, c in enumerate(selected_state_list):
    #         for y_trace, label in zip(y_data, labels):
    #     # labeling the left_side of the plot
    #             annotations.append(dict(xref='paper', x=0.001, y=df.loc[c][0],
    #                                           xanchor='right', yanchor='middle',
    #                                           text= c,
    #                                           font=dict(family='Arial',
    #                                                     size=16),
    #                                           showarrow=False))
    #     # labeling the right_side of the plot
    #     annotations.append(dict(xref='paper', x=0.95, y=df.loc[c].iloc[-1],
    #                             xanchor='left', yanchor='middle',
    #                             text='{:.0f}%'.format(df.loc[c].iloc[-1]*100),
    #                             font=dict(family='Arial',
    #                                       size=16),
    #                             showarrow=False))

    # fig.update_layout(annotations=annotations)

    return fig




# the output's component_property was 'children'

# # State use for click action required!
#
# # if you want to change by input, then plotting should be in the def at the end ("input_data")
# # if you only want to show specific graph, then can plot it in the app.layout
#
# ## Multiple dropdown list
# def goplot(button_1, country_selection1, country_selection2, country_selection3, country_selection4):
#     # make sure it won't generate blank plot by itself and show error
#     if not (country_selection1, country_selection2, country_selection3, country_selection4):
#         return no_update  # This is prevent the web run the function without any input
#     # elif button_1:
#     #     outputtt = create_trend_line_infection_rate_2day(infection_data, country_selection1 + country_selection2
#     #                                                      + country_selection3 + country_selection4,
#     #                                                      '<b>Infection Rate Per Country</b>')
#     else:
#         outputtt = create_trend_line_infection_rate_2day(infection_data, country_selection1 + country_selection2
#                                                          + country_selection3 + country_selection4,
#                                                          '<b>Infection Rate Per Country</b>')
#     return outputtt
# return country_selection


#### test multiple buttons
# State use for click action required!
# if you want to change by input, then plotting should be in the def at the end ("input_data")
# if you only want to show specific graph, then can plot it in the app.layout

@app.callback(
    Output(component_id='output-graph1', component_property='figure'),
    [Input('button_alpha', 'n_clicks'), Input('button_order', 'n_clicks')],
)

def goplot1(button_alpha, button_order):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button_order' in changed_id:
        case_array, case_array_std, date_list, state_list = case_data_process_tt(df_case_pop)
        output1 = create_case_heatmap(case_array, case_array_std, date_list, state_list)
    elif 'button_alpha' in changed_id:
        case_array, case_array_std, date_list, state_list = case_data_process(df_case_pop)
        output1 = create_case_heatmap(case_array, case_array_std, date_list, state_list)
    else:
        case_array, case_array_std, date_list, state_list = case_data_process(df_case_pop)
        output1 = create_case_heatmap(case_array, case_array_std, date_list, state_list)
    return output1

@app.callback(
    Output(component_id='output-graph2', component_property='figure'),
    [Input('button_info', 'n_clicks'), Input('button_phase', 'n_clicks')],
)

def goplot2(button_info, button_phase):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button_info' in changed_id:
        output2 = create_real_map()
    elif 'button_phase' in changed_id:
        output2 = create_real_map(phase=True)
    else:
        output2 = create_real_map()
    return output2

@app.callback(
    Output(component_id='output-graph3', component_property='figure'),
    # [Input('button1', 'n_clicks'), Input('button2', 'n_clicks')],
    [Input('button', 'n_clicks')],
    [State(component_id='state_selection', component_property='value')],
)

def goplot3(button, state_selection):
    # make sure it won't generate blank plot by itself and show error
    if not state_selection:
        return no_update  # This is prevent the web run the function without any input
    else:
        output3 = create_trend_line(df_case_std_row, state_selection)
    return output3


# n_clicks is required for click event
if __name__ == '__main__':
    app.run_server(debug=True)

import script
from script.ETL import *

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

#### Pipeline ####

df = get_data()
state_pop_dict = make_state_pop_dict()

## heat map
df_case = moving_avg_df(ETL_df(df), death=False)
df_case_pop = data_process_avg(df_case, state_pop_dict)
case_array, case_array_std, date_list, state_list = case_data_process(df_case_pop)

## real map
# phase_dict = get_phase_dict() # outdated
df_map = get_news_link_df(df_map_ETL(df, make_df_map(df_case_pop), df_case_pop, state_pop_dict))

## individual line
df_case_std_row = pd.DataFrame(standardized_row(case_array), index=state_list)

## other info
current_date = date_list[-1].strftime('%Y/%m/%d')



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
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
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
                            y=1.007,  # paper 1
                            xanchor='right', yanchor='middle',
                            text='✦: Date of peak number of new cases within the state',
                            font=dict(family='Arial', color='gray',
                                      size=15), showarrow=False))
    fig.update_layout(annotations=annotations)
    return fig


def create_real_map(mask=False):
    if mask is False:
        fig = go.Figure(data=go.Choropleth(
            locations=df_map['abbr'],
            z=df_map['case'],
            text=df_map['text'],  # hover info
            locationmode='USA-states',
            colorscale='Reds',
            autocolorscale=False,
            marker_line_color='white',  # line markers between states
            # colorbar_title="Daily Cases per 1M",
            colorbar=dict(
                title='Daily Cases per 1M',
                thickness=10, len=0.5),
            hovertemplate='%{text}' + '<extra></extra>',
        ))
    else:
        color_list = [px.colors.qualitative.Vivid[i] for i in [3, 2, 10]]  # [3, 2, 10, 6]
        mask_status_list = ['Mandatory', 'Sometimes Required', 'Not Required']
        color_dict = {p: c for c, p in zip(color_list, mask_status_list)}

        fig = px.choropleth(data_frame=df_map,
                            locations=df_map['abbr'],
                            color=df_map['mask'],
                            scope="usa", locationmode="USA-states",
                            color_discrete_map=color_dict,
                            # hover_data=['text_phase'],
                            hover_data={'abbr': False, 'state': True, 'mask': True, },
                            labels={'mask': 'Masks'}
                            )
        fig.update_traces(marker_line_width=0.2, marker_opacity=0.9)

    fig.update_layout(
        title={'text': '',  # f'Latest Daily Information({current_date})'
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
            't': 10,  # 20
            'b': 10,
        },
    )
    return fig


def create_trend_line(df, selected_state_list):
    color_n = len(selected_state_list)
    cm = px.colors.qualitative.T10
    color_m = cm[0:color_n]

    fig = go.Figure()
    fig.layout.template = 'ggplot2'
    for i, c in enumerate(selected_state_list):
        fig.add_trace(go.Scatter(x=date_list, y=df.loc[c].values * 100,
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
            y=[df.loc[c].values[-1] * 100],
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
            'x': 0.01,
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
            'fixedrange': True,
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
            'showticklabels': False,
            'fixedrange': True, },
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

    ## Adding labels # outdated
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


#### Layout #####

####  drop down list function
state_options = [dict(label=s, value=s) for s in state_list]
dropdown_function_state = dcc.Dropdown(id='state_selection',
                                       options=state_options,
                                       multi=True,
                                       value=['New York', 'Arizona'],
                                       placeholder="Pick states",
                                       # className="dcc_control"
                                       )

#### Initiate the app ####
## style setting
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['https://unpkg.com/tailwindcss@1.5.1/dist/tailwind.min.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID, dbc.themes.BOOTSTRAP],
                # meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
                )
server = app.server
app.title = 'Curves in the US'

## Wireframe
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div(
                    'Spotting the Curves in the US',
                    style={'font-size': '40px', 'font-style': 'extra bold',
                           # 'color': 'gray',
                           }
                ),
                # dcc.Markdown(
                #     '''
                #     A Better Overview and Handy Dashboard of COVID-19 in Each State of the United States
                #     ''',
                #     style={'font-style': 'italic', 'display': 'inline'}),
                html.Div([html.Div('A Better Overview of COVID-19 in Each State of the United States',
                                   style={'font-style': 'italic', 'display': 'inline'}),
                          html.Div(f'Data updated daily (Last Updated: {current_date})',
                                   style={'font-style': 'italic', 'color': 'blue', 'display': 'inline',
                                          'paddingLeft': '10px'}),
                          ]),
                # html.Div(f'Data updated daily (Latest Date: {current_date})',
                #         style={'font-style': 'italic', 'color': 'blue', 'display': 'inline', 'paddingLeft': '0px'}),
                # dcc.Markdown(
                #     f'''
                #     Data updated daily (Latest Date: {current_date})
                #     ''',
                #     style={'font-style': 'italic', 'color': 'blue', 'display': 'inline'})
            ]),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.H4('Purpose and Features', style={'font-style': 'bold'}),
                dcc.Markdown(
                    '''
                To better understand COVID-19 in the US, this dashboard presents an overview of state-level data 
                with complementary information. This dashboard aims to show individual trends within each state, 
                extract timely state-level phase information and **provide a consistent and easy-to-read platform 
                other than media outlets**. \n 
                
                The major metrics include:
                - ```7-Day Moving Average```: A series of averages of daily increases in cases or deaths over time 
                (applied to all numerical data in the heat map and the choropleth map below)
                - ```Mask Mandate```: The mandate of masks issued by each state government (as reported
                 by *The New York Times*; scroll to the bottom of this page for a list of the latest news and info about
                  each state)
                '''
                ),  ## and filter the most important data from the news in an easy-to-read, straightforward platform.
                html.H4('References and Relevant Readings', style={'font-style': 'bold', "margin-top": "10%"}),  # 30px
                html.Div(dcc.Markdown(
                    '''
                    The state-level data is from [The New York Times GitHub](https://github.com/nytimes/covid-19-data). 
                    The information about the phases of each state is retrieved from 
                    [The New York Times](https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html).
                    Here is another [country-level dashboard](https://spot-the-curve-coronavirus.herokuapp.com/) and 
                    a [Medium article](https://towardsdatascience.com/spot-the-curve-visualization-of-cases-data-on-coronavirus-8ec7cc1968d1?source=friends_link&sk=4f984ca1c1e4df9535b33d9ccab738ee) 
                    about interpreting Coronavirus data. If you notice any mistakes or have any comments regarding the 
                    data, visualizations or dashboard, please feel free to contact the author 
                    (*[Jeff Lu](https://www.linkedin.com/in/jefflu-chia-ching-lu/)*) 
                ''',
                ),
                    style={'font-size': '13px', 'color': 'grey'}
                ),
            ], width=5, style={'paddingRight': '3%'}  # 50px
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
            ], width=7, style={'paddingLeft': '3%'}), ],
        ),
        dbc.Row([
            dbc.Col(html.H4('Daily New Cases per 1M Residents', style={'font-style': 'bold'}),
                    # class='col-xl-9 col-lg-8 col-sm-12 col-xs-12'
                    width='col-xl-9 col-lg-8 col-sm-6 col-xs-4',  # 5
                    ),
            dbc.Col([
                html.Div(
                    dbc.Button('Alphabetical', id='button_alpha', outline=True, color="primary", style={}),
                ),
            ], style={'padding': '0px 5px'}
                # width={"size": 1, "offset": 5},
            ),
            dbc.Col([
                html.Div(
                    dbc.Button('Ranked by Peak Date', id='button_order', outline=True, color="primary", style={}),
                ),
            ], style={'padding': '0px 5px'}
                # width={"size": 1, "offset": 0},
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
            dbc.Col(html.H4(f'Latest Info ({current_date})', style={'font-style': 'bold'}),
                    width='col-xl-9 col-lg-8 col-sm-6 col-xs-4'
                    ),
            dbc.Col([
                html.Div(
                    dbc.Button('Daily Cases & Deaths', id='button_info', outline=True, color="primary", style={}),
                    # margin-left': '50px'
                ),
            ], style={'padding': '0px 5px'}  # width={"size": 1, "offset": 5}
            ),
            dbc.Col([
                html.Div(
                    dbc.Button('State Mask Mandate', id='button_mask', outline=True, color="primary", style={}),
                    # 'margin-left': '0px'
                ),
            ], style={'padding': '0px 5px'}, )  # width={"size": 1, "offset": 0}),
        ], style={'paddingTop': '70px'}),
        dbc.Row([
            dbc.Col([
                html.Div(
                    dcc.Graph(id='output-graph2', figure=create_real_map(mask=True), animate=None)  # create_real_map()
                    # dcc.Graph(id='output-graph2', animate=None) # create_real_map()
                ),
            ]),
        ], ),
        dbc.Row(dbc.Col(html.H5('''Latest Daily Data and Relevant News''', style={'font-style': 'bold'}),
                        width='col-xl-9 col-lg-8 col-sm-6 col-xs-4'
                        ), style={'paddingTop': '50px', 'paddingBottom': '10px'}),
        html.Div(dash_table.DataTable(id='datatable',
                                      data=df_map.to_dict('records'),
                                      columns=[{"id": "state", "name": [""], "presentation": "markdown"},
                                               {"id": "case", "name": ["Daily Cases per 1M"],
                                                "presentation": "markdown", },
                                               {"id": "death", "name": ["Daily Deaths per 1M"],
                                                "presentation": "markdown", },
                                               # {"id": "phase", "name": ["State's Phase"], "presentation": "markdown"},
                                               {"id": "link", "name": ["Relevant News"], "presentation": "markdown", }],
                                      page_action='none',
                                      style_table={'height': '300px',
                                                   # 'width': '1000px',
                                                   'width': '90%',
                                                   'minWidth': '90%',
                                                   'overflowY': 'auto', },
                                      # fixed_columns={'headers': True, 'data': 5}, #this will limit the display
                                      style_as_list_view=True,
                                      style_cell={
                                          # 'padding': '0px',
                                          # 'width': '50px',
                                          'textAlign': 'left'},
                                      style_cell_conditional=[{'if': {'column_id': 'state'}, 'textAlign': 'left'},
                                                              {'if': {'column_id': 'case'}, 'textAlign': 'left'}, ],
                                      #  https://github.com/plotly/dash-table/issues/777 bug still exists!
                                      style_header={
                                          # 'backgroundColor': '#F7FBFE',
                                          'fontWeight': 'bold',
                                          'font-size': 16,
                                          'font-family': 'Arial'
                                      },
                                      css=[{'selector': '.row', 'rule': 'margin: 0'}]
                                      ),
                 style={'width': '100%',  # 100
                        # 'display': 'flex', # seems like this!
                        'justify-content': 'center'
                        }
                 ),
    ], fluid=True,
        style={'paddingLeft': '7%', 'paddingRight': '7%', 'paddingBottom': '3%', 'paddingTop': '3%'})
])


## test multiple buttons
# State use for click action required!
# if you want to change by input, then plotting should be in the def at the end ("input_data")
# if you only want to show specific graph, then can plot it in the app.layout

## Callback

@app.callback(
    Output(component_id='output-graph1', component_property='figure'),
    [Input('button_alpha', 'n_clicks'), Input('button_order', 'n_clicks')],
)
def goplot1(button_alpha, button_order):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button_order' in changed_id:
        case_array, case_array_std, date_list, state_list = case_data_process(df_case_pop, order=True)
        output1 = create_case_heatmap(case_array, case_array_std, date_list, state_list)
    elif 'button_alpha' in changed_id:
        case_array, case_array_std, date_list, state_list = case_data_process(df_case_pop, order=False)
        output1 = create_case_heatmap(case_array, case_array_std, date_list, state_list)
    else:
        case_array, case_array_std, date_list, state_list = case_data_process(df_case_pop, order=True)
        output1 = create_case_heatmap(case_array, case_array_std, date_list, state_list)
    return output1


@app.callback(
    Output(component_id='output-graph2', component_property='figure'),
    # Input('button_info', 'n_clicks'),
    [Input('button_info', 'n_clicks'), Input('button_mask', 'n_clicks')],
)
def goplot2(button_info, button_mask):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button_info' in changed_id:
        output2 = create_real_map()
    elif 'button_mask' in changed_id:
        output2 = create_real_map(mask=True)
    else:
        output2 = create_real_map()
    return output2

## outdated
# def goplot2(button_info):
#     changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
#     output2 = create_real_map()
#     return output2

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

#### NOTE ####
#### Alarm function (TBD)
# find out which states break their state records in the last 5 days (meaning highly dangerous states!)
# def alarm_state_list():
#     dict_alarm = (df_case_pop.idxmax().sort_values() > df_case_pop.tail(5).index[0]).to_dict()
#     alarm_state_list = [k for k, v in dict_alarm.items() if v == True]
#     return alarm_state_list
# alarm_state_list = alarm_state_list()


## official doc
## https://plotly.com/python/reference/
## https://dash.plotly.com/dash-html-components

## HOVER AND HIGHLIGHT
# https://stackoverflow.com/questions/53327572/how-do-i-highlight-an-entire-trace-upon-hover-in-plotly-for-python
# https://plotly.com/javascript/plotlyjs-events/

## COOL WAY TO DO THE SIMILAR THING. CAN LEARN MARKDOWN FROM THIS
# https://github.com/COVID19Tracking/covid-tracking-dash/blob/master/covid_tracking/app.py
# http://35.212.27.3:8050/

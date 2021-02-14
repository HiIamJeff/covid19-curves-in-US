
# import requests
import pandas as pd
import numpy as np

## Load the data
def get_data():
    data_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
    data_set = pd.read_csv(data_url, dtype={'fips': 'category', 'state': 'category'})
    return data_set

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
    return df


def moving_avg_df(df, death=True):
    if death is False:
        df_case = df.pivot_table(values='cases', index=['date'], columns='state').rolling(window=7).mean().diff().iloc[
                  7:]
    else:
        df_case = df.pivot_table(values='deaths', index=['date'], columns='state').rolling(window=7).mean().diff().iloc[
                  7:]
    # replace some negative value with 0
    df_case = df_case.applymap(lambda x: 0 if x < 0 else x)
    return df_case


def make_state_pop_dict():
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


def case_data_process(df_case_pop, order=False):
    if order: ## rank by higest cases date
        state_list = df_case_pop.idxmax().sort_values().index
    else:
        state_list = df_case_pop.columns

    ## cases data
    case_array = np.array([df_case_pop[state] for state in state_list])

    ## scale with all states (find the smallest daily increase and biggest daily increase; use this as scale)
    # max: 590.443, min: 0
    def standardized_all(array):
        return (array - np.min(array, axis=0)) / np.ptp(array, axis=None)

    case_array_std = standardized_all(case_array)  # max: 590.443, min: 0

    ## date
    date_list = list(dict.fromkeys(df_case_pop.index))  # check!

    return case_array, case_array_std, date_list, state_list

### Real map
def make_df_map(df_case_pop):
    df_case_pop2 = pd.DataFrame(df_case_pop.iloc[-1])
    df_case_pop2.columns = ['case']
    df_case_pop2 = df_case_pop2.reset_index()
    return df_case_pop2


def create_mask_dict():
    from bs4 import BeautifulSoup
    import requests
    url = "https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'lxml')
    # mask
    sr1 = soup.select('.g-masks , .g-name a')

    state_list = []
    mask_list = []
    for i, n in enumerate(sr1):
        if i % 2 == 0:
            state_list.append(n.text[:-2])
        else:
            mask_list.append(n.text)

    mask_list = [' '.join(w.title().split()[1:]) for w in mask_list] # remove excessive words, capitalize char
    mask_dict = {s: m for s, m in zip(state_list, mask_list) if s not in ['Washington, D.C.', 'Puerto Rico']}
    return mask_dict


def df_map_ETL(df, df_map, df_case_pop, state_pop_dict):
    # death data with most recent records (proccessed with 7-day average)
    death_dict = moving_avg_df(ETL_df(df), death=True).iloc[-1].to_dict()
    death_dict = {s: round(death_dict.get(s) / state_pop_dict.get(s) * 1000000, 3) for s in df_case_pop.columns}
    mask_dict = create_mask_dict()

    # abbreviate data for plotting function
    import us
    abbr_dict = us.states.mapping('name', 'abbr')

    # pipeline
    df_map['abbr'] = df_map['state'].map(abbr_dict)
    df_map['mask'] = df_map['state'].map(mask_dict)
    df_map['death'] = round(df_map['state'].map(death_dict).astype(float), 2)
    df_map['case'] = round(df_map['case']).astype(int)
    # for hoverinfo
    df_map['text'] = (
            '<I><b>' + df_map['state'].astype('str') + '</b></I><br>' + 'New Cases per 1M Resident: '
            + df_map['case'].astype('str')
            + '<br>' + 'New Deaths per 1M Resident: ' + df_map['death'].astype('str') + '<extra></extra>')
    return df_map

## css select
# state name and mask rule
# .g-masks , .g-name a
# state name, mask rule and link (but has other garbage, need to clean)
# .g-masks , a


def get_news_link_df(df_map):
    from bs4 import BeautifulSoup
    import requests
    import re

    url = "https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'lxml')

    # state
    sr1 = soup.select('.g-name')
    scrap_list1 = [i.text[:-2] for i in sr1]

    # link
    sr2 = soup.select('.g-link a', attrs={'href': re.compile('^http://')})
    scrap_list2 = [link.get('href') for link in sr2]
    link_list = [f'''[News Link]({l})''' for l in scrap_list2]
    df_link = pd.DataFrame(np.array([scrap_list1, link_list]).T, index=scrap_list1, columns=['state', 'link'])
    df_link.drop(['Washington, D.C.', 'Puerto Rico'], inplace=True)
    df_link = pd.merge(df_link, df_map, on='state')
    return df_link

### individual line
def standardized_row(array):
    return ((array.T - np.min(array.T, axis=0)) / np.ptp(array, axis=1)).T


print('## ETL script loaded ##')


## outdated
# def get_phase_dict():
#     from bs4 import BeautifulSoup
#     import requests
#     url = "https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html"
#     res = requests.get(url)
#     #     print(res)
#     # the number 200 is http status code
#
#     soup = BeautifulSoup(res.text, 'lxml')
#     sr = soup.select(".g-name , .g-cat-subhed span")
#     # scrap_list = [i.text for i in sr] # bug fix
#     scrap_list = [i.text[:-2] for i in sr]
#     scrap_list = [x for x in scrap_list if x not in ['Washington, D.C.', 'Puerto Rico']]
#     # ['Washington, D.C.', 'Puerto Rico']
#     import us
#     scrap_set = set(scrap_list)
#     state_set = set([s.name for s in us.states.STATES])
#     phase_set = scrap_set - state_set
#
#     phase_dict = {}
#     for i in scrap_list:
#         if i in phase_set:
#             phase_n = i
#             continue
#         phase_dict[i] = phase_n
#     return phase_dict
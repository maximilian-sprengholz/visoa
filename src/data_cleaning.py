# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# try setting time
import locale
try:
    locale.setlocale(locale.LC_TIME, 'de_DE')
    de_time = True
except:
    de_time = False

# read data from repo (direct link was not working)
df = pd.read_csv('./data/raw/Bundesländer-Atlas - Institutionen.csv')

# subset
df_temporal = df[[
    # standard items
    'Unnamed: 0', 'Name der Institution', 'Ort', 'Einrichtungsart',
    'Postleitzahl vor Ort GERiT', 'DESTATIS Fächerguppe GERiT',
    # OA Dummies (not all have corresponding dates)
    'OA Webseite der Institution', 'OA Webseite Datum',
    'OA-Beauftragte/r', 'OA-Beauftragte/r URL', 'OA-Beauftragte/r URL Datum',
    'Ansprechperson für Open Access', 'Ansprechperson für Open Access URL', 'Ansprechperson für Open Access URL Datum',
    'Repositorium URL',
    'Publikationsfonds',
    'OA-Verlag', 'OA-Verlag URL', 'OA-Verlag Datum Recherche',
    'OJS-Standort', 'OJS-Standort URL', 'OJS-Standort Datum Recherche',
    'OA/OS-Preis URL',
    # Ratification/Implementation dates
    'OA Policy', 'OA Policy Datum',
    'OA Leitlinie', 'OA Leitlinie Datum',
    'Berliner Erklärung', 'Datum Berliner Erklärung (Bundesländeratlas)',
    'OA2020', 'Datum OA2020 (Bundesländeratlas)',
    # last update (default when no date info)
    'aktualisiert am'
]].copy()
df_temporal.rename(columns = {'Unnamed: 0':'Bundesland'}, inplace = True)

# Merge data about Traegerschaft
df_traeger = pd.read_csv('./data/raw/Bundesländer-Atlas_Institutionen_Trägerschaft_Hochschulen.csv', usecols=[0,1,2], sep=';', encoding = 'latin1')
df_temporal = df_temporal.merge(df_traeger, how='left', on=['Bundesland', 'Name der Institution'])

# rename
df_temporal.columns = [
        # standard items
        'state', 'name', 'city', 'type', 'plz', 'discipline',
        # OA Dummies (not all have corresponding dates)
        'oa_website_url', 'oa_website_rdate',
        'oa_representative_name', 'oa_representative_url', 'oa_representative_rdate',
        'oa_contact_name', 'oa_contact_url', 'oa_contact_rdate',
        'oa_repo_url',
        'oa_fund',
        'oa_publisher_name', 'oa_publisher_url', 'oa_publisher_rdate',
        'oa_hosting_name', 'oa_hosting_url', 'oa_hosting_rdate',
        'oa_award_url',
        # Ratification/Implementation dates
        'oa_policy', 'oa_policy_date',
        'oa_leitlinie', 'oa_leitlinie_date',
        'oa_berlin_erkl', 'oa_berlin_erkl_date',
        'oa_oa2020', 'oa_oa2020_date',
        # last update (default when no date info)
        'lastupdate',
        # Traegerschaft
        'traegerschaft'
]

df_temporal['oa_representative_url'].value_counts(dropna=False)
df_temporal['oa_contact_url'].value_counts(dropna=False)

# date cleaning
'''
Done in multiple steps corresponding to the formats in the dataset.
'''

# loop over all datevars and apply conversion for present formats
formats = ['%d.%m.%Y', '%m.%Y', '%b %Y', '%B %Y', '% Y']
datevars = ['oa_website_rdate', 'oa_representative_rdate', 'oa_contact_rdate',
            'oa_publisher_rdate', 'oa_hosting_rdate', 'oa_policy_date',
            'oa_leitlinie_date', 'oa_berlin_erkl_date', 'oa_oa2020_date', 'lastupdate']
# For Kaggle: change dates to US english in subsequent loop (not necessary when locale is 'de_DE')
if (de_time == False):
    transl_b = {
        '(?i)Mär ': 'Mar ',
        '(?i)Mai ': 'May ',
        '(?i)Okt ': 'Oct ',
        '(?i)Dez ': 'Dec '
        }
    transl_B = {
        '(?i)Januar ': 'January ',
        '(?i)Februar ': 'February ',
        '(?i)März ': 'March ',
        '(?i)Juni ': 'June ',
        '(?i)Juli ': 'July ',
        '(?i)Oktober ': 'October ',
        '(?i)Dezember ': 'December '
        }
for i in datevars:
    # cleaned target
    df_temporal[i + '_cleaned'] = None
    df_temporal[i + '_cleaned'] = df_temporal[i + '_cleaned'].astype('datetime64[ns]')
    # convert datestrings
    for j in formats:
        # minor destructive cleanup
        df_temporal['temp'] = df_temporal[i].copy()
        if (de_time == False):
            df_temporal['temp'] = df_temporal['temp'].replace(regex=transl_b)
            df_temporal['temp'] = df_temporal['temp'].replace(regex=transl_B)
        df_temporal['temp'] = df_temporal.temp.str.replace(',','.')
        df_temporal['temp'] = df_temporal.temp.str.replace('ß','0')
        # convert Excel style VALUE dates to standard format (weird stuff)
        df_temporal['xl_corr'] = df_temporal['temp'].replace(regex=r'.*(\.|[a-zA-Z]).*', value=np.nan) # replace all but numbers
        df_temporal['xl_corr'] = df_temporal['temp'].replace(regex=r'\b[0-9][0-9][0-9][0-9]\b', value=np.nan) # replace if Y only
        df_temporal['xl_corr'] = pd.to_numeric(df_temporal['xl_corr'], errors='coerce')
        df_temporal['xl_corr'] = pd.TimedeltaIndex(df_temporal['xl_corr'], unit='d') + dt.datetime(1899, 12, 30)
        # dates
        df_temporal['temp'] = pd.to_datetime(df_temporal['temp'], format=j, errors='coerce')
        df_temporal.loc[df_temporal['xl_corr'].isna() == False, 'temp'] = df_temporal['xl_corr']
        # fill values into cleaned var
        df_temporal.loc[df_temporal['temp'].isna() == False, i + '_cleaned'] = df_temporal['temp']
    # keep only cleaned var
    df_temporal.drop(i, axis=1, inplace=True)
    df_temporal.rename(columns = {i + '_cleaned': i}, inplace = True)
df_temporal.drop(['temp', 'xl_corr'], axis=1, inplace=True)

### Build and clean non-date indicators
'''
Basic cleaning. Possible errors not accounted for:
- Typos (so that regex does not find the words/groups)
- NA != 0
'''

# Create dummies
dummyvars = ['oa_website', 'oa_representative', 'oa_contact', 'oa_repo', 'oa_fund', 'oa_publisher',
             'oa_hosting', 'oa_award', 'oa_policy', 'oa_leitlinie', 'oa_berlin_erkl', 'oa_oa2020']
bool_replace = {
        r'(?i).*\b(ja)\b.*': '1',
        r'(?i).*\b(nein)\b.*': '0',
        r'(?i).*\b(NA)\b.*': '0',
        np.nan: '0',
        r'(?i).*\b(recherche fehlt)\b.*': np.nan,
        }

for i in dummyvars:
    if i in df_temporal.columns:
        # replace when yes/no column already present
        df_temporal[i] = df_temporal[i].replace(regex=bool_replace)
        # special case OA Leitlinie (no URL field, but URL values)
        if (i == 'oa_leitlinie'):
            df_temporal[i] = df_temporal[i].replace(regex=r'.*\..*', value='1') # all other strings containing a period
    else:
        # fetch from url info otherwise
        df_temporal[i] = df_temporal[i + '_url']
        df_temporal[i] = df_temporal[i].replace(regex=bool_replace)
        df_temporal.loc[(df_temporal[i].isna() == False) & (df_temporal[i] != '0'), i] = '1'
    # convert to boolean
    df_temporal[i] = pd.to_numeric(df_temporal[i], errors='coerce')
    df_temporal[i] = df_temporal[i].astype('Int64')

# Create rdate values based on lastupdate if not present
for i in ['oa_website', 'oa_representative', 'oa_contact', 'oa_repo', 'oa_fund', 'oa_publisher', 'oa_hosting', 'oa_award']:
    if i + '_rdate' not in df.columns:
        df_temporal[i + '_rdate'] = df_temporal['lastupdate']

### Further date cleaning

# Problem: ratification date missing but item True
print('Ratification date missing but item True')
print('Policy ' + str(df_temporal.loc[(df_temporal['oa_policy_date'].isna()) & (df_temporal['oa_policy'] != 0), 'name'].count()))
print('Leitlinie ' + str(df_temporal.loc[(df_temporal['oa_leitlinie_date'].isna()) & (df_temporal['oa_leitlinie'] != 0), 'name'].count()))
print('Berliner Erklaerung ' + str(df_temporal.loc[(df_temporal['oa_berlin_erkl_date'].isna()) & (df_temporal['oa_berlin_erkl'] != 0), 'name'].count()))
print('OA2020 ' + str(df_temporal.loc[(df_temporal['oa_oa2020_date'].isna()) & (df_temporal['oa_oa2020'] != 0), 'name'].count()))

# Date Leitlinie Hochschule Anhalt is wrong -> replace
df_temporal.loc[(df_temporal['name']=='Hochschule Anhalt'), 'oa_leitlinie_date'] = pd.Timestamp('2021-03-17')

# Replace policy date with date of last update when missing, but create flag before to allow for visual indication
for i in ['oa_policy', 'oa_leitlinie', 'oa_berlin_erkl', 'oa_oa2020']:
    df_temporal.loc[(df_temporal[i + '_date'].isna()) == False & (df_temporal[i] == 1), i + '_date_flag'] = 0
    df_temporal.loc[(df_temporal[i + '_date'].isna()) & (df_temporal[i] != 0), i + '_date_flag'] = 1
    df_temporal.loc[(df_temporal[i + '_date'].isna()) & (df_temporal[i] != 0), i + '_date'] = df_temporal['lastupdate']
    df_temporal[i + '_date_flag'] = df_temporal[i + '_date_flag'].astype('Int64')

# Merge OA Policy and Leitlinie Items, use whatever was first; carry over missing date flag value
df_temporal['oa_pol_leit'] = df_temporal['oa_policy'].fillna(0) + df_temporal['oa_leitlinie'].fillna(0)
df_temporal.loc[(df_temporal['oa_policy'].isna()) & (df_temporal['oa_leitlinie'].isna()), 'oa_pol_leit'] = np.nan # distinguish between 0 and missing
df_temporal['oa_pol_leit_date'] = df_temporal.loc[(df_temporal['oa_pol_leit'] == 1) & (df_temporal['oa_policy']==1), 'oa_policy_date'] # policy only
df_temporal.loc[(df_temporal['oa_pol_leit'] == 1) & (df_temporal['oa_leitlinie']==1), 'oa_pol_leit_date'] = df_temporal['oa_leitlinie_date'] # leitlinie only
df_temporal.loc[(df_temporal['oa_pol_leit'] == 2) & (df_temporal['oa_policy_date'] < df_temporal['oa_leitlinie_date']), 'oa_pol_leit_date'] = df_temporal['oa_policy_date'] # policy first
df_temporal.loc[(df_temporal['oa_pol_leit'] == 2) & (df_temporal['oa_leitlinie_date'] <= df_temporal['oa_policy_date']), 'oa_pol_leit_date'] = df_temporal['oa_leitlinie_date'] # leitlinie first
df_temporal['oa_pol_leit_date_flag'] = 0
df_temporal.loc[(df_temporal['oa_pol_leit'] == 1) & (df_temporal['oa_policy_date_flag']==1), 'oa_pol_leit_date_flag'] = 1 # policy only
df_temporal.loc[(df_temporal['oa_pol_leit'] == 1) & (df_temporal['oa_leitlinie_date_flag']==1), 'oa_pol_leit_date_flag'] = 1 # leitlinie only
df_temporal.loc[(df_temporal['oa_policy_date_flag']==1) & (df_temporal['oa_leitlinie_date_flag']==1), 'oa_pol_leit_date_flag'] = 1 # leitlinie only
df_temporal['oa_pol_leit'] = df_temporal['oa_pol_leit'].replace(2,1)

# Merge representative & contact
df_temporal['oa_rep_con'] = df_temporal['oa_representative'].fillna(0) + df_temporal['oa_contact'].fillna(0)
df_temporal['oa_rep_con'] = df_temporal['oa_rep_con'].replace(2,1)

# Check relevant plot data
df_plot = df_temporal.loc[(df_temporal['type'] != 'Hochschule') | (df_temporal['traegerschaft'].isin(['kirchlich, staatlich anerkannt', 'öffentlich-rechtlich']))]
dummyvars = ['oa_website', 'oa_rep_con', 'oa_repo', 'oa_fund', 'oa_publisher',
             'oa_hosting', 'oa_award', 'oa_berlin_erkl', 'oa_oa2020', 'oa_pol_leit']
print('Indicator distribution (sample does not contain private Hochschulen)')
for i in dummyvars:
    print(i)
    print(df_plot[i].value_counts())

print('No BE, Policy, Leitlinie, or OA2020:')
print(df_plot.loc[
    (df_plot['oa_berlin_erkl']==0)
    & (df_plot['oa_pol_leit']==0)
    & (df_plot['oa_oa2020']==0), 'name'].count())

print('No BE, Policy, Leitlinie, or OA2020, but at least one OA infrastructure Item true:')
print(df_plot.loc[
    (df_plot['oa_berlin_erkl']==0)
    & (df_plot['oa_pol_leit']==0)
    & (df_plot['oa_oa2020']==0)
    & (
        (df_plot['oa_website']==1)
        | (df_plot['oa_rep_con']==1)
        | (df_plot['oa_repo']==1)
        | (df_plot['oa_fund']==1)
        | (df_plot['oa_publisher']==1)
        | (df_plot['oa_hosting']==1)
    ), 'name'].count())

# plot
fig = go.Figure()
clist = ['rgba(51,187,238,100)', 'rgba(0,153,136,100)', 'rgba(238,119,51,100)', 'rgba(238,51,119,100)']
tlist = ['Berliner Erklärung', 'OA Policy/Leitlinie', 'OA2020']
for i, j in enumerate(['oa_berlin_erkl_date', 'oa_pol_leit_date', 'oa_oa2020_date']):
    fig.add_trace(
        go.Box(
            x = df_plot[j],
            whiskerwidth=0,
            pointpos = 0,
            marker = dict(color = clist[i]),
            line = dict(color = 'rgba(0,0,0,0)'),
            fillcolor = 'rgba(0,0,0,0)',
            boxpoints = 'all',
            name = tlist[i] + ' ',
            customdata = df_temporal['name'],
            hovertemplate = '<b>%{customdata}</b><br>%{x}<extra></extra>',
        )
    )
fig.update_layout(
    width = 1000,
    height = 600,
    title = 'Date of ratification/adoption by institution'
)
fig.show()

# reorder and export
firstcols = ['name', 'state', 'city', 'plz', 'type', 'traegerschaft', 'discipline']
col_order = df_plot.columns.drop(firstcols).tolist()
col_order.sort()
col_order = firstcols + col_order
df_plot = df_plot[col_order]
df_plot.to_csv('./data/processed/visoa_temp_cleaned.csv')

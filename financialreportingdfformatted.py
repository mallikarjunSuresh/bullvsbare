import pandas as pd
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests


# This will keep tickers + gics industries & sub industries
def save_sp500_stocks_info():
    print("Getting SP500 stocks info from wikipedia")
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    stocks_info=[]
    tickers = []
    securities = []
    gics_industries = []
    gics_sub_industries = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        security = row.findAll('td')[1].text
        gics_industry = row.findAll('td')[3].text
        gics_sub_industry = row.findAll('td')[4].text

        tickers.append(ticker.lower().replace("\n", ""))
        securities.append(security)
        gics_industries.append(gics_industry.lower())
        gics_sub_industries.append(gics_sub_industry.lower())

    stocks_info.append(tickers)
    stocks_info.append(securities)
    stocks_info.append(gics_industries)
    stocks_info.append(gics_sub_industries)

    stocks_info_df = pd.DataFrame(stocks_info).T
    stocks_info_df.columns=['tickers','security','gics_industry','gics_sub_industry']
    stocks_info_df['seclabels'] = 'SP500'
    stocks_info_df['labels'] = stocks_info_df[['tickers','security', 'gics_industry','gics_sub_industry','seclabels']].apply(lambda x: ' '.join(x), axis=1)

    # Create a list of dict based on tickers and labels
    dictlist = []
    for index, row in stocks_info_df.iterrows():
        dictlist.append({'value':row['tickers'], 'label':row['labels']})
    return dictlist


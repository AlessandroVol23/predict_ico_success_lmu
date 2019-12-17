
from src.features.utils import read_processed_data
import pandas as pd
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
import re
import requests  

url_reg_ex =  regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def _make_valid_url( url):

    result = re.match(url_reg_ex, url)

    if result:
        return url
    else:
        return "https://"+url


def _check_http_status_code(url):
    if url == "NAN":
        return 0
    url = _make_valid_url(url)
    try:
        r = requests.head(url)
        if r.status_code >= 200 and r.status_code < 300:
            return 1
        else:
            return 0
    except requests.ConnectionError:
        return 0

def _transform_link_binary( column="links_homepage", na_strategy ="set:NAN"):
    df_bitcoin, df, df_test, df_gem_btc_usd = read_processed_data()
    df_copy = df.copy()
    results = []
    if na_strategy.find(":") != -1:
        strat = na_strategy.split(":")
        if strat[0] == "set":
            df_copy.loc[df_copy[column].isna(), column] = strat[1]
        else:
            raise ValueError("Unrecognized command strategy for {column}")

    with Pool(processes=20) as pool:
        results = pool.map(_check_http_status_code, df_copy[column])

    df_copy["links_homepage_available"] = results

    df_copy.to_csv('data/raw/1_training_data_sets/1_training_data.csv')

from src.features.utils import read_processed_data
import pandas as pd
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
import re
import requests
from tqdm import tqdm

url_reg_ex = regex = re.compile(
    r'^(?:http|ftp)s?://'  # http:// or https://
    # domain...
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def _make_valid_url(url):

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
        r = requests.head(url, timeout=3)
        if r.status_code >= 200 and r.status_code < 300:
            return 1
        else:
            return 0
    except requests.ConnectionError:
        return 0
    except requests.Timeout:
        return 0
    except:
        return 0


def transform_link_binary(column="links_homepage", na_strategy="set:NAN"):
    df_bitcoin, df, df_test, df_gem_btc_usd, df_gem_eth_usd, df_gem_ltc_usd, df_icobench = read_processed_data()
    df_copy = pd.concat([df, df_test], sort=True)
    results = []
    if na_strategy.find(":") != -1:
        strat = na_strategy.split(":")
        if strat[0] == "set":
            df_copy.loc[df_copy[column].isna(), column] = strat[1]
        else:
            raise ValueError("Unrecognized command strategy for {column}")

    with Pool(processes=20) as pool:
        results = list(tqdm(pool.imap(_check_http_status_code,
                                      df_copy[column]), total=len(df_copy[column])))

    df_new = df_copy[['OBS_ID']].copy()
    df_new["links_homepage_available"] = results
    df_new.to_csv(
        'data/external/Links_Homepage_available.csv', index=None)

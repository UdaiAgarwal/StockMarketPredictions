# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:34:09 2021

@author: udaia

This script is to get data from the alpha vantage api
"""


import pandas as pd
import numpy as np
import requests
import json
import re
import os



# ----- Get API KEY for AlphaVantage -----
CWD = os.getcwd()
API_KEY = ""
with open(CWD + "/Utility/key.json") as f:
    keys = json.load(f)
    
API_KEY = keys["APIKEY"]
# -----

# ----- API FUNCTIONS -----
default_url = "https://www.alphavantage.co/query?function={FUNC}&apikey={API_KEY}"

Search = "SYMBOL_SEARCH"
TimeSeriesDaily = "TIME_SERIES_DAILY"
GlobalQuote = "GLOBAL_QUOTE"
Intraday = "TIME_SERIES_INTRADAY"

# -----


# ----- UTILITY FUNCTIONS -----
def search(search_word):
    """
    Get the Symbol for the company you're looking for.
    search_word: Key term to search for
    """
    url = default_url.format(FUNC=Search, API_KEY=API_KEY)
    url += "&keywords={KEYWORD}".format(KEYWORD=search_word)
    r = requests.get(url)
    try:
        data = pd.DataFrame(r.json()["bestMatches"])
    except:
        return None

    data.columns = data.columns.str[3:]
    data = data[data["currency"] == "INR"]
    data = data[["symbol", "name", "type"]]
    return data


def get_time_series_daily(symbol, name):
    """
    Get Time series data on a daily basis.
    symbol: Acceptable symbol for company
    """
    url = default_url.format(FUNC=TimeSeriesDaily, API_KEY=API_KEY)
    url += "&symbol={SYMBL}&outputsize=full".format(SYMBL=symbol)
    r = requests.get(url)
    try:
        data = pd.DataFrame(r.json()["Time Series (Daily)"]).T
    except:
        return None
    
    data.columns = data.columns.str[3:]
    data = data.reset_index()
    data = data.rename(columns = {"index": "date"})
    data["symbol"] = symbol
    data["name"] = name
    return data


def get_time_series_daily_compact(symbol, name):
    """
    Get Time series data on a daily basis.
    symbol: Acceptable symbol for company
    name: Name of the company
    """
    url = default_url.format(FUNC=TimeSeriesDaily, API_KEY=API_KEY)
    url += "&symbol={SYMBL}".format(SYMBL=symbol)
    r = requests.get(url)
    try: 
        data = pd.DataFrame(r.json()["Time Series (Daily)"]).T
    except:
        return None
    
    data.columns = data.columns.str[3:]
    data = data.reset_index()
    data = data.rename(columns = {"index": "date"})
    data["symbol"] = symbol
    data["name"] = name
    return data


def get_quote(symbol):
    """
    Get Quote price with OHCL values.
    symbol: Acceptable symbol for company
    """
    url = default_url.format(FUNC=GlobalQuote, API_KEY=API_KEY)
    url += "&symbol={SYMBL}".format(SYMBL=symbol)
    r = requests.get(url)
    try: 
        data = pd.DataFrame(r.json()).T.reset_index()
    except:
        return None 
    
    data.drop(columns="index", inplace=True)
    data.columns = data.columns.str[3:]
    return data
# -----


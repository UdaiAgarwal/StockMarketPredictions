# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:45:08 2021

@author: udaia
"""

from Utility.AlphaVantage import *
import os
from tqdm import tqdm
import pandas as pd
from glob import glob



CWD = os.getcwd()

# ----- Companies to look for -----
with open(CWD + "/Utility/symbols.json") as f:
    symbols = json.load(f)
    
company_symbols = list(symbols.keys()) # Companies
# -----

# ----- Existing files -----
existing = pd.Series(list(glob(CWD + "/Data/*.csv")))
existing = existing.apply(lambda x: x.split("\\")[-1][:-4])
# -----


counter = 0
for c in company_symbols:
    print("Fetchnig data for: ", c, end="")
    if(symbols[c] in list(existing)):
        print("(Already exists)")
        continue
    else:
        print()
    
    # Break after every 5 files
    if(counter == 5):
        time.sleep(5)
        counter = 0
    
    df = get_time_series_daily(c, symbols[c])
    try:
        df.to_csv("{CWD}/Data/{NAME}.csv".format(CWD=CWD, NAME=symbols[c]), index = False)
    except:
        print("Data Not Present for ", c)
    
    counter +=1 
        


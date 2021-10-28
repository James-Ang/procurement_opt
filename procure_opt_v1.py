# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:20:39 2019

@author: Acceval Pte Ltd
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

df = pd.read_excel('C:\\Users\\Acceval Pte Ltd\\.spyder-py3\\Smart_Tradzt\\procurement_opt\\data_source.xlsx',sheet_name='Sheet1')
hist_price = pd.read_excel('C:\\Users\\Acceval Pte Ltd\\.spyder-py3\\Smart_Tradzt\\procurement_opt\\data_source.xlsx',sheet_name='Sheet2')
hist_price.date = [i.toordinal() for i in hist_price.date] #converting date to ordinal
#df.loc[1,:]
#hist_price.min(axis=0)

#hist_price_A = np.array([723,702,688,685,700,721,749])
#hist_price_B = np.array([748,727,713,710,725,746,774])
#hist_price_spot = np.array([686,686,627,627,631,631,668])

ind_min_A =  hist_price["Sup_A"].idxmin()
ind_min_B =  hist_price["Sup_B"].idxmin()
date_min_A = hist_price["date"][ind_min_A]

plt.plot(hist_price.date,hist_price["Sup_A"],'ro:')
plt.grid(b=True)

WACC = 0.1  # (%)
daily_cons = 100    # (MT)
plan_buc = 28   # (days)
tank_top = 3000 # (MT)
tank_heel = 400   # (MT)

"""Delivery lead time"""
lead_time_A = 2 #days   (domestic)
lead_time_B = 10 #days   (import)
lead_time_spot = 10 #days

""" Max Delivery Volume """
max_sup_A = 250
max_sup_B = 300
max_sup_spot = 300

OI_ini = 800
daily_consumption = [100]
num_days = 7

x

for i in range(1,num_days+1):
    print(i)
    
    

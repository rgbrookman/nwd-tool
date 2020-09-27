import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import pearsonr
import pylab as pl
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import ezsheets
import gspread
from gspread_pandas import Spread, Client
%matplotlib inline 

gc = gspread.service_account(filename='newnwd.json') 

def Richard():
    nwd_rb = gc.open_by_key('1WkGJy-yaqawIib5AGZKFFpfm007vEGoUIBGxiJwVDQ4')
    spread_rb = Spread('Richards NoWastedDays')
    rb_nwd = nwd_rb.get_worksheet(2)
    rb_df = pd.DataFrame(rb_nwd.get_all_records())
    rb_t1 = rb_df['1TO']
    rb_df['1TO'] = pd.to_numeric(rb_t1, errors='coerce')
    rb_t2 = rb_df['2TO']
    rb_df['2TO'] = pd.to_numeric(rb_t2, errors='coerce')
    rb_t3 = rb_df['3TO']
    rb_df['3TO'] = pd.to_numeric(rb_t3, errors='coerce')
    rb_t4 = rb_df['4TO']
    rb_df['4TO'] = pd.to_numeric(rb_t4, errors='coerce')
    rb_t5 = rb_df['5TO']
    rb_df['5TO'] = pd.to_numeric(rb_t5, errors='coerce')
    rb_t6 = rb_df['Feeling_Value']
    rb_df['Feeling_Value'] = pd.to_numeric(rb_t6, errors='coerce')
    rb_t7 = rb_df['Task_Completion_Rate']
    rb_df['Task_Completion_Rate'] = pd.to_numeric(rb_t7, downcast='float')
    rb_t8 = rb_df['DAILY_SCORE']
    rb_df['DAILY_SCORE'] = pd.to_numeric(rb_t8, errors='coerce')
    rb_df['Date'] = rb_df['Date'].astype('datetime64')
    rb_t9 = rb_df['Bank_Bal_Index']
    rb_df['Bank_Bal_Index'] = pd.to_numeric(rb_t9, errors='coerce')
    rb_t10 = rb_df['RMA']
    rb_df['RMA'] = pd.to_numeric(rb_t10, downcast='float')
    rb_t11 = rb_df['Bank_Bal']
    rb_df['Bank_Bal'] = pd.to_numeric(rb_t11, errors='coerce')
    values = {'1TO': 0,'2TO': 0,'3TO': 0,'4TO': 0,'5TO': 0}
    rb_df[['1TO','2TO','3TO','4TO','5TO']].fillna(value=values)
    rb_df['day_of_week'] = rb_df['Date'].dt.day_name()
    rb_df['month_of_year'] = rb_df['Date'].dt.month_name()
    rb_df['year'] = pd.DatetimeIndex(rb_df['Date']).year
    rb_downum = {'Monday': 1, 'Tuesday': 2,'Wednesday': 3,'Thursday': 4, 'Friday': 5,'Saturday': 6,'Sunday': 7}
    rb_df['downum'] = rb_df['day_of_week'].replace(rb_downum)
    rb_moynum = {'January': 1,'February': 2,'March': 3,'April': 4,'May': 5, 'June': 6,'July': 7,'August': 8,'September': 9,'October': 10,'November': 11,'December': 12}
    rb_df['moynum'] = rb_df['month_of_year'].replace(rb_moynum)
    rb_df['TP'] = rb_df['Feeling_Value'] + (rb_df['Task_Completion_Rate'] * 5)
    rb_bins_tcr = np.linspace(min(rb_df['Task_Completion_Rate']),max(rb_df['Task_Completion_Rate']),6)
    rb_group_names = ["<20%","20-40","40-60","60-80","80-100"] 
    rb_df['CR_CAT'] = pd.cut(rb_df['Task_Completion_Rate'],rb_bins_tcr,labels=rb_group_names,include_lowest=True) 
    rb_df['1-2'] = rb_df['1TO'] + rb_df['2TO']
    rb_df['1-3'] = rb_df['1TO'] + rb_df['2TO'] + rb_df['3TO']
    rb_df['1-4'] = rb_df['1TO'] + rb_df['2TO'] + rb_df['3TO'] + rb_df['4TO']
    rb_df['1-5'] = rb_df['1TO'] + rb_df['2TO'] + rb_df['3TO'] + rb_df['4TO'] + rb_df['5TO']
    rb_df['SCORE_CAT'] = pd.qcut(rb_df['DAILY_SCORE'], 4)
    rb_df['PREV_SCORE'] = rb_df['DAILY_SCORE'].shift(1, axis = 0) 
    rb_df['NEXT_SCORE'] = rb_df['DAILY_SCORE'].shift(-1, axis = 0) 
    rb_df['SCORE_SHIFT'] = rb_df['DAILY_SCORE'] - rb_df['PREV_SCORE']
    rb_df['SS_COPY'] = rb_df['SCORE_SHIFT']
    rb_df.loc[rb_df.SS_COPY < 0, 'SS_COPY'] = 0
    rb_df.loc[rb_df.SS_COPY > 0, 'SS_COPY'] = 1
    rb_df['SS_COPY'] = rb_df['SS_COPY'].fillna(0)
    rb_df['plus_minus'] = rb_df['SS_COPY']
    rb_df = rb_df.drop(['SS_COPY'], axis=1)
    rb_df['plus_minus'].astype(int)
    rb_df['PREV_FEEL'] = rb_df['Feeling_Value'].shift(1, axis = 0) 
    rb_df['NEXT_FEEL'] = rb_df['Feeling_Value'].shift(-1, axis = 0) 
    rb_df['FEEL_SHIFT'] = rb_df['Feeling_Value'] - rb_df['PREV_FEEL']
    rb_df['FEEL_INDEX'] = rb_df['Feeling_Value'] / rb_df['Feeling_Value'].mean()
    rb_df['FEEL_SCALE'] = rb_df['Feeling_Value'] / rb_df['Feeling_Value'].max() 
    rb_bins_feel = np.linspace(min(rb_df['Feeling_Value']),max(rb_df['Feeling_Value']),6)
    rb_df['FEEL_CAT'] = pd.cut(rb_df['Feeling_Value'],rb_bins_feel,labels=rb_group_names,include_lowest=True) 
    rb_df['PREV_TCR'] = rb_df['Task_Completion_Rate'].shift(1, axis = 0) 
    rb_df['NEXT_TCR'] = rb_df['Task_Completion_Rate'].shift(-1, axis = 0) 
    rb_df['TCR_SHIFT'] = rb_df['Task_Completion_Rate'] - rb_df['PREV_TCR']  
    rb_df['TCR_INDEX'] = rb_df['Task_Completion_Rate'] / rb_df['Task_Completion_Rate'].mean()
    rb_df['TCR_SCALE'] = rb_df['Task_Completion_Rate'] / rb_df['Task_Completion_Rate'].max() 
    rb_df['TC_TOTAL'] = rb_df[['1TO','2TO','3TO','4TO','5TO','1-2','1-3','1-4','1-5']].max(axis=1)
    rb_df['TC_TOTAL'] = rb_df['TC_TOTAL'].fillna(0)
    rb_df['PREV_TC'] = rb_df['TC_TOTAL'].shift(1, axis = 0) 
    rb_df['NEXT_TC'] = rb_df['TC_TOTAL'].shift(-1, axis = 0) 
    rb_df['TC_SHIFT'] = rb_df['TC_TOTAL'] - rb_df['PREV_TC']  
    rb_df['TC_INDEX'] = rb_df['TC_TOTAL'] / rb_df['TC_TOTAL'].mean()
    rb_df['TC_SCALE'] = rb_df['TC_TOTAL'] / rb_df['TC_TOTAL'].max()
    rb_df['TC_ZSCORE'] = (rb_df['TC_TOTAL'] - rb_df['TC_TOTAL'].mean()) / rb_df['TC_TOTAL'].std()
    rb_bins_tc = np.linspace(min(rb_df['TC_TOTAL']),max(rb_df['TC_TOTAL']),6)
    rb_df['TC_CAT'] = pd.cut(rb_df['TC_TOTAL'],rb_bins_tc,labels=rb_group_names,include_lowest=True) 
    rb_df['TRUE_PROD'] = rb_df['TC_TOTAL'] + (rb_df['Task_Completion_Rate'] * 5)
    rb_df['PREV_TP'] = rb_df['TRUE_PROD'].shift(1, axis = 0) 
    rb_df['NEXT_TP'] = rb_df['TRUE_PROD'].shift(-1, axis = 0) 
    rb_df['TP_SHIFT'] = rb_df['TRUE_PROD'] - rb_df['PREV_TP']  
    rb_df['TP_INDEX'] = rb_df['TRUE_PROD'] / rb_df['TRUE_PROD'].mean()
    rb_df['TP_SCALE'] = rb_df['TRUE_PROD'] / rb_df['TRUE_PROD'].max() 
    rb_bins_tp = np.linspace(min(rb_df['TRUE_PROD']),max(rb_df['TRUE_PROD']),6)
    rb_df['TP_CAT'] = pd.cut(rb_df['TRUE_PROD'],rb_bins_tp,labels=rb_group_names,include_lowest=True) 
    rb_df['Week_Num'] = rb_df['Date'].dt.week
    df_1_rb = rb_df[rb_df['Day_Input']>=1] 
    df_0_rb = rb_df[rb_df['Day_Input']<=0]
    df_1_rb = df_1_rb.drop(['D_ID','D_ID_Check','Day_Input'], axis=1)
    df_0_rb = df_0_rb.drop(['D_ID','D_ID_Check','Day_Input'], axis=1) 
    spread_rb.df_to_sheet(df_1_rb, index=False, sheet='Your Data', start='A1', replace=True)
    
    Richard()

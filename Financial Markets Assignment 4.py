#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import datetime
import statsmodels.api as sm


# In[3]:


os.chdir('/Users/duanyihong/Desktop/financial markets/Assignment 4/equity transaction cost_data')
data=pd.read_csv('merged.csv')
del data['Unnamed: 0']
data


# In[4]:


#Compute the benchmark
data['benchmark']=(data['bid']+data['ask'])/2
#calculate the transaction costs for buyers
buy=data[data['BuySell']=='B']
stocks=set(buy['symbol'])  #extract all the stocks 
#first get the transaction costs at each transaction time
buy['Costs']=(buy['Price']-buy['benchmark'])/buy['benchmark']
stocks=set(buy['symbol'])  #extract all the stocks 
#then get the transaction costs at each transaction time
buy['Costs']=(buy['Price']-buy['benchmark'])/buy['benchmark']


# In[5]:


#calculate the value-weighted average transaction costs for every stock

def AvgCost(df):
    SumCost=sum(df['Shares']*df['Costs'])
    avg=SumCost/sum(df['Shares'])
    return avg
    
BuyCost=pd.DataFrame(buy.groupby('symbol').apply(AvgCost))#set the column name as 'buy_cost'
BuyCost.columns = ['buy_cost']


# In[6]:


#The same methods for the stocks 'S'
Sell=data[data['BuySell']=='S']
Sell['Costs']=(Sell['benchmark']-Sell['Price'])/Sell['benchmark']
SellCost=pd.DataFrame(Sell.groupby('symbol').apply(AvgCost))
SellCost.columns = ['sell_cost']

#Merge the buy cost and sell cost
Cost=pd.merge(BuyCost,SellCost,left_index=True,right_index=True)


# In[7]:


Cost


# In[8]:


Mktcap=pd.read_csv('mktcap.csv')
#take the logarithm of the market cap
import math
#First transform the type of market cap to float type
Mktcap['mktcap']=Mktcap['mktcap'].apply(float)
Mktcap['log_mkt']=np.log(Mktcap['mktcap'])


# In[9]:


#Build a dataframe to store the buy and sell regression results
Transac_mkt=pd.DataFrame(columns=['buy','sell'],index=['intercept','beta'])
#Define a regression function
def reg(d1,d2):
    d2=sm.add_constant(d2.values)
    model=sm.OLS(d1.values,d2).fit()
    return model.params[0],model.params[1]


# In[10]:


#regress the buy and sell costs on market capitalization
Transac_mkt['buy']=reg(Cost['buy_cost'],Mktcap['log_mkt'])
Transac_mkt['sell']=reg(Cost['sell_cost'],Mktcap['log_mkt'])


# In[11]:


Transac_mkt


# In[12]:


#View the summary of regression of buy_cost
y=Cost['buy_cost'].values
x=Mktcap['log_mkt'].values
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()


# In[13]:


#View the summary of sell cost
y=Cost['sell_cost'].values
x=Mktcap['log_mkt'].values
x=sm.add_constant(x)
model=sm.OLS(y,x).fit()
model.summary()


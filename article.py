#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:26:35 2019

@author: duanyihong
"""
import tushare as ts
import pandas as pd
import numpy as np 
import os

os.chdir('/Users/duanyihong/Desktop/stocks')
#Pstocks里有975只股票
Pstocks=pd.DataFrame(pd.read_csv('Pstocks.csv'))
Pstocksgr=Pstocks.groupby(Pstocks['Stkcd'])
#计算日收益率
def dreturn(df):
    df['dreturn']=df['Clsprc'].shift(1)/df['Clsprc']-1
    return df['dreturn']
Stocksreturn=Pstocksgr.apply(dreturn)
Stocksreturn=Stocksreturn.fillna(0)
#计算累计收益率
def cumreturn(df):
    df['cumreturn']=np.cumprod(1+df['dreturn'])
    return df['cumreturn']
Stocksreturn=pd.DataFrame(Stocksreturn)
Stocksreturn['cumreturn']=np.cumprod(1+Stocksreturn['dreturn'])

Prcfirst=pd.DataFrame(Pstocksgr.head(1)['Clsprc'])
Stocksreturngr=Stocksreturn.groupby('Stkcd')
Extractcumreturn=Stocksreturngr.tail(1)['cumreturn']
Cumreturn=np.array(list(Extractcumreturn))
Prc=np.array(list(Prcfirst['Clsprc']))
Mom=pd.DataFrame(np.multiply(Prc,Cumreturn))

universe=ts.get_hs300s()
df1=pd.DataFrame(pd.read_excel('TRD_Dalyr.xlsx'))
df1=df1.drop([0,1])
df2=pd.DataFrame(pd.read_excel('TRD_Dalyr1.xlsx'))
df2=df2.drop([0,1])
df3=pd.DataFrame(pd.read_excel('TRD_Dalyr2.xlsx'))
df3=df3.drop([0,1])

df1=df1.set_index(df1['Trddt'])
df2=df2.set_index(df2['Trddt'])
df3=df3.set_index(df3['Trddt'])
stock1=list(df1[df1.index=='2009-01-05']['Stkcd'].values)
stock2=list(df2[df2.index=='2009-01-05']['Stkcd'].values)
stock3=list(df3[df3.index=='2010-04-01']['Stkcd'].values)
stock=stock1+stock2+stock3
universe=list(universe['code'].values)
stocku=[i for i in stock if i in universe]    #提取出了在股票池中的股票代码

stocksinfo=pd.concat([df1,df2,df3])

stocku=pd.DataFrame(stocku)

stocksinfo=stocksinfo.sort_index(by=['Stkcd','Trddt'])
stocksinfo[stocksinfo['Stkcd']==stocku]


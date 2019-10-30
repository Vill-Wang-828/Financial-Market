#!/usr/bin/env python
# coding: utf-8

# In[197]:


import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# In[2]:


os.getcwd()


# In[3]:


os.chdir('/Users/duanyihong/Desktop/financial markets')


# In[54]:


df_emon=pd.read_csv('equal_weighted_returns_monthly.csv')
df_eyear=pd.read_excel('equal-weighted returns.xlsx')
df_vmon=pd.read_csv('value-weighted returns-monthly.csv')
df_vyear=pd.read_excel('value-weighted returns-annually.xlsx')
df_ffmon=pd.read_csv('monthly FF-3 factors.csv')
df_ffyear=pd.read_excel('annual FF-3 factors.xlsx')


# In[62]:


df_mon_e=pd.merge(df_emon,df_ffmon,on='dates')


# In[82]:


df_vmon=df_vmon.dropna()
df_vmon['dates']=df_vmon['dates'].astype(int)


# In[83]:


df_mon_v=pd.merge(df_vmon,df_ffmon,on='dates')


# In[251]:


portfolios=['Lo 10', '2-Dec',
       '3-Dec', '4-Dec', '5-Dec', '6-Dec', '7-Dec', '8-Dec', '9-Dec', 'Hi 10']


# In[287]:


#create a dataframe to store regression results
capm=pd.DataFrame(columns=['Annualized monthly mean return','beta'],index=portfolios)


# In[294]:


def beta(data1,data2):
    data2=sm.add_constant(data2)
    model=sm.OLS(data1,data2).fit()
    return model.params[1]
def Ann_Return(data1):
    return data1.values.mean()*12


# In[297]:


for i in portfolios:
    capm['beta'][i]=beta(df_mon_v[i],df_mon_v['Mkt-RF'])
    capm['Annualized monthly mean return'][i]=Ann_Return(df_mon_v[i])


# In[300]:


#plotting
plt.xlabel('beta')
plt.ylabel('Average Annualized Monthly Returns(%)')
plt.title('1963-2019')
plt.scatter(capm['beta'].values,capm['Annualized monthly mean return'])
x=np.linspace(0,2,100)
y=df_mon_v['RF'].mean()*12+df_mon_v['Mkt-RF'].mean()*12*x
plt.plot(x,y,'-',color='blue')
plt.show()


# In[301]:


#divide the samples
df_mon_v['dates']=df_mon_v['dates'].astype(int)
sample1=df_mon_v[(df_mon_v['dates']>=196307)&(df_mon_v['dates']<=199012)]
sample2=df_mon_v[df_mon_v['dates']>199012]


# In[302]:


capmsample1=pd.DataFrame(columns=['Annualized monthly mean return','beta'],index=portfolios)


# In[303]:


for i in portfolios:
    capmsample1['beta'][i]=beta(sample1[i],sample1['Mkt-RF'])
    capmsample1['Annualized monthly mean return'][i]=Ann_Return(sample1[i])


# In[306]:


plt.xlabel('beta')
plt.ylabel('Average Annualized Monthly Returns(%)')
plt.title('1963-1990')
plt.scatter(capm['beta'].values,capm['Annualized monthly mean return'])
x=np.linspace(0,2,100)
y=df_mon_v['RF'].mean()*12+df_mon_v['Mkt-RF'].mean()*12*x
plt.plot(x,y,'-',color='purple')
plt.show()


# In[305]:


capmsample2=pd.DataFrame(columns=['Annualized monthly mean return','beta'],index=portfolios)
for i in portfolios:
    capmsample2['beta'][i]=beta(sample2[i],sample2['Mkt-RF'])
    capmsample2['Annualized monthly mean return'][i]=Ann_Return(sample2[i])


# In[307]:


plt.xlabel('beta')
plt.ylabel('Average Annualized Monthly Returns(%)')
plt.title('1990-2019')
plt.scatter(capm['beta'].values,capm['Annualized monthly mean return'])
x=np.linspace(0,2,100)
y=df_mon_v['RF'].mean()*12+df_mon_v['Mkt-RF'].mean()*12*x
plt.plot(x,y,'-',color='green')
plt.show()


# In[159]:


df_ffmon=df_ffmon.rename(columns={'dates':'ym'})


# In[347]:


GM_f=pd.merge(GM,df_ffmon)
KO_f=pd.merge(KO,df_ffmon)
MSFT_f=pd.merge(MSFT,df_ffmon)


# In[348]:


GM_f['retx']=GM_f['retx'].fillna(method='ffill')  #fill the NaN using the following value
GM_f['Ri-RF']=GM_f['ret']-GM_f['RF']
GM_f=GM_f.dropna()


# In[323]:


KO_f=KO_f.dropna()  #fill the NaN using the following value
KO_f['Ri-RF']=KO_f['ret']-KO_f['RF']


# In[324]:


MSFT_f=MSFT_f.dropna()  #fill the NaN using the following value
MSFT_f['Ri-RF']=MSFT_f['ret']-MSFT_f['RF']
stock=['GM','KO','MSFT']


# In[309]:


stocks_iid=pd.DataFrame(columns=['alpha','beta','std_alpha','std_beta'],index=stock)


# In[331]:


def Alpha_beta(data1,data2):
    data2=sm.add_constant(data2)
    model=sm.OLS(data1,data2).fit()
    return model.params[0],model.params[1],model.params[0]/model.tvalues[0],model.params[1]/model.tvalues[1]


stocks_iid.iloc[0]=Alpha_beta(GM_f['Ri-RF'],GM_f['Mkt-RF'])
stocks_iid.iloc[1]=Alpha_beta(KO_f['Ri-RF'],KO_f['Mkt-RF'])
stocks_iid.iloc[2]=Alpha_beta(MSFT_f['Ri-RF'],MSFT_f['Mkt-RF'])
stocks_iid


# In[341]:


#compute the standard error for the estimated coefficients
#assumption of heteroskedasticity(white standard errors)
def Alpha_beta_white(data1,data2):
    data2=sm.add_constant(data2)
    model=sm.OLS(data1,data2).fit()
    return model.params[0],model.params[1],model.get_robustcov_results().HC0_se[0],model.get_robustcov_results().HC0_se[1]

stocks_white=pd.DataFrame(columns=['alpha','beta','std_alpha','std_beta'],index=stock)
stocks_white.iloc[0]=Alpha_beta_white(GM_f['Ri-RF'],GM_f['Mkt-RF'])
stocks_white.iloc[1]=Alpha_beta_white(KO_f['Ri-RF'],KO_f['Mkt-RF'])
stocks_white.iloc[2]=Alpha_beta_white(MSFT_f['Ri-RF'],MSFT_f['Mkt-RF'])
stocks_white


# In[346]:


#assumption for serial-correlation----newey west standard error
stocks_newey=pd.DataFrame(columns=['alpha','beta','std_alpha','std_beta'],index=stock)
import statsmodels
import math
def Alpha_beta_newey(data1,data2):
    data2=sm.add_constant(data2)
    model=sm.OLS(data1,data2).fit()
    return model.params[0],model.params[1],math.sqrt(statsmodels.stats.sandwich_covariance.cov_hac(model,nlags=4)[0,0]),math.sqrt(statsmodels.stats.sandwich_covariance.cov_hac(model,nlags=4)[1,1])
stocks_newey.iloc[0]=Alpha_beta_newey(GM_f['Ri-RF'],GM_f['Mkt-RF'])
stocks_newey.iloc[1]=Alpha_beta_newey(KO_f['Ri-RF'],KO_f['Mkt-RF'])
stocks_newey.iloc[2]=Alpha_beta_newey(MSFT_f['Ri-RF'],MSFT_f['Mkt-RF'])
stocks_newey


#!/usr/bin/env python
# coding: utf-8

# In[835]:


import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col


# In[39]:


#change the working directory
os.chdir('/Users/duanyihong/Desktop/financial markets/Assignment 2')


# In[40]:


Industry=pd.DataFrame(pd.read_excel('10 industry-MonthV.xlsx'))
Sample_industry=Industry[(Industry['dates']>=196301)&(Industry['dates']<=201812)]


# In[803]:


Mom=pd.DataFrame(pd.read_excel('MomentumMonthly.xlsx'))
Mom_sample=Mom[(Mom['dates']>=196301)&(Mom['dates']<=201812)]


# In[804]:


FF3=pd.DataFrame(pd.read_csv('F-F_Research_Data_factors.csv'))
FF3_sample=FF3[(FF3['dates']>=196301)&(FF3['dates']<=201812)]


# In[45]:


df1=pd.merge(Sample_industry,Mom_sample,on='dates')
data=pd.merge(df1,FF3_sample,on='dates')


# In[97]:


Industries=['NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops',
       'Hlth ', 'Utils', 'Other']


# In[73]:


CAPM=pd.DataFrame(columns=['alpha','beta','std_alpha','std_beta'],index=Industries)


# In[58]:


#define the function to calculate the alpha and beta and their iid standard errors
def Alpha_beta(df1,df2):
    df2=sm.add_constant(df2)
    model=sm.OLS(df1,df2).fit()
    return model.params[0],model.params[1],model.params[0]/model.tvalues[0],model.params[1]/model.tvalues[1]


# In[688]:


for i in range(len(Industries)):
    CAPM.iloc[i]=Alpha_beta(data[Industries[i]]-data['RF'],data['Mkt-RF'])


# In[689]:


CAPM  #CAPM beta alpha and iid 


# In[68]:


#Now compute the white std and newey std
import math
CAPM_adj=pd.DataFrame(columns=['White std_alpha','White std_beta','newey std_alpha','newey std_beta'],index=Industries)
def sterr(df1,df2):
    df2=sm.add_constant(df2)
    model=sm.OLS(df1,df2).fit()
    return model.get_robustcov_results().HC0_se[0],model.get_robustcov_results().HC0_se[1],math.sqrt(statsmodels.stats.sandwich_covariance.cov_hac(model,nlags=6)[0,0]),math.sqrt(statsmodels.stats.sandwich_covariance.cov_hac(model,nlags=6)[1,1])


# In[77]:


for i in range(len(Industries)):
    CAPM_adj.iloc[i]=sterr(data[Industries[i]]-data['RF'],data['Mkt-RF'])


# In[78]:


CAPM_adj  #CAPM with white std and newey std


# In[89]:


#regression for Fama-French three-factor model and Carhart
FF3=pd.DataFrame(columns=['alpha','beta_mkt','beta_Size','beta_BM','std_alpha','std_betamkt','std_betasize','std_betaBM'],index=Industries)
for i in range(len(Industries)):
    FF3.iloc[i]=Alpha_beta(data[Industries[i]]-data['RF'],data[['Mkt-RF','SMB','HML']])


# In[90]:


FF3   #the iid std of alpha and beta under FF3 regression


# In[83]:


#the white std and newey std for FF3 model
FF3_adj=pd.DataFrame(columns=['White std_alpha','White std_beta','newey std_alpha','newey std_beta'],index=Industries)
for i in range(len(Industries)):
    FF3_adj.iloc[i]=sterr(data[Industries[i]]-data['RF'],data[['Mkt-RF','SMB','HML']])


# In[84]:


FF3_adj    #the adjusted FF3 model


# In[100]:


#Let's do Carhart four-factor model
Carh4=pd.DataFrame(columns=['alpha','beta','std-alpha','std-beta'],index=Industries)
for i in range(len(Industries)):
    Carh4.iloc[i]=Alpha_beta(data[Industries[i]]-data['RF'],data[['Mom   ','Mkt-RF','SMB','HML']])


# In[101]:


Carh4    #the carhart 4 factor model with iid standard error


# In[102]:


#as for the white and newey std
Carh4_adj=pd.DataFrame(columns=['white std-alpha','white std-beta','newey std-alpha','newey std-beta'],index=Industries)
for i in range(len(Industries)):
    Carh4_adj.iloc[i]=sterr(data[Industries[i]]-data['RF'],data[['Mom   ','Mkt-RF','SMB','HML']])


# In[103]:


Carh4_adj    #carhart 4-factor model for white and newey std


# In[705]:


#For Fama-Macbeth regression
#First load the data and restrict time period from 1963 to 2018
IndusFull=pd.DataFrame(pd.read_csv('30_Industry_Portfolios.csv'))
IndusFull=IndusFull.dropna()
IndusFull['dates']=IndusFull['dates'].astype(int)
IndusFull_sample=IndusFull[(IndusFull['dates']>=196301)&(IndusFull['dates']<=201812)]


# In[706]:


RF=data[['dates','RF']]   #process the monthly estimates of RF
Data=pd.merge(IndusFull_sample,RF)   #merge the 30 industries data and RF
Industry30=['Food ', 'Beer ', 'Smoke', 'Games', 'Books', 'Hshld', 'Clths',
       'Hlth ', 'Chems', 'Txtls', 'Cnstr', 'Steel', 'FabPr', 'ElcEq', 'Autos',
       'Carry', 'Mines', 'Coal ', 'Oil  ', 'Util ', 'Telcm', 'Servs', 'BusEq',
       'Paper', 'Trans', 'Whlsl', 'Rtail', 'Meals', 'Fin  ', 'Other']


# In[707]:


#dealing with LogSize
#first load the size and BM data and restrict them to a specific time period


# In[759]:


#take the log of size factors
LSize=np.log(Size)   #take the log of size 
#Now deal with Book-to-market ratio
d={'dates':IndusFull_sample['dates']}
BM_M=pd.DataFrame(data=d)
#write a function to extract the first four digits of an integer
def digit(x):
    x=str(x)
    x=x[0:4]
    return int(x)
for name in Industry30:
    value=np.zeros(len(IndusFull_sample['dates']))
    for i in range(len(IndusFull_sample['dates'])):
        value[i]=BM[name][digit(IndusFull_sample['dates'].iloc[i]-12)-1963+37]
    BM_M[name]=value
BM_M=BM_M.set_index('dates')


# In[760]:


LSize_cro=LSize.transpose()
BM_cro=BM_M.transpose()


# In[761]:


#create the risk premium matrix:
Datapre=Data.astype(float)
for i in Industry30:
    Datapre[i]=Datapre[i]-Datapre['RF']
CroSec=Datapre[Industry30].transpose()
#the first step of Fama-macbeth regression-run the regression for each time period
Datapre['dates']=Datapre['dates'].astype(int)
def BetaT(df1,df2):
        df2=sm.add_constant(df2)
        model=sm.OLS(df1,df2).fit()
        return model.params.values


# In[819]:


Time=Datapre['dates']
Beta=pd.DataFrame(columns=['alpha','beta_BM','beta_size'],index=BM_cro.columns)
#BetaT(CroSec['196303'],pd.concat([BM_cro[196303],LSize_cro[196303]],axis=1))
CroSec.columns=BM_cro.columns
for i in range(len(BM_cro.columns)-1):
    Beta.iloc[i]=BetaT(CroSec[Time[i+1]]/100,pd.concat([BM_cro[Time[i]],LSize_cro[Time[i]]],axis=1))


# In[820]:


Beta    #beta from time 0 to time T
Beta=Beta.dropna()
FM=pd.DataFrame(columns=['values'],index=['alpha','beta_BM','beta_size'])
FM.iloc[0]=Beta['alpha'].mean()
FM.iloc[1]=Beta['beta_BM'].mean()
FM.iloc[2]=Beta['beta_size'].mean()


# In[821]:


Beta   #fama-macbeth regression results


# In[822]:


FM


# In[830]:


FM


# In[834]:


y=CroSec[196302]/100
X=pd.concat([BM_cro[Time[0]],LSize_cro[Time[0]]],axis=1)
X=sm.add_constant(X)
model=sm.OLS(y,X).fit()
model.summary()


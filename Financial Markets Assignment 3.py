#!/usr/bin/env python
# coding: utf-8

# In[6]:


#For Question 2
import pandas as pd
import numpy as np
import os
import math
import statsmodels.api as sm


# In[7]:


os.chdir('/Users/duanyihong/Desktop/financial markets/Assignment 3')
#read the datafile
df=pd.read_csv('Performancemeasures.csv')


# In[8]:


#a. Interpret the 
#1. conduct a univariate regression of excess return on market excess return
y=df['Ln/Sh Eq Hedge Fund USD']-df['RF']
X=df['Mkt-RF']
X=sm.add_constant(X)
CAPM=sm.OLS(y,X).fit()
print(CAPM.summary())


# In[9]:


#a multivariate regression on market size, value and momentum factors
X=df[['Mkt-RF','SMB','HML','UMD']]
X=sm.add_constant(X)
MULTI=sm.OLS(y,X).fit()
print(MULTI.summary())


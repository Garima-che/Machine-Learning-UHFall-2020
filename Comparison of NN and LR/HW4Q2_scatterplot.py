#!/usr/bin/env python
# coding: utf-8

# # Online News Shares

# ## Importing the libraries

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Importing the dataset

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
df = pd.read_csv('OnlineNewsPopularity.csv')
df.head(5)


# In[7]:


df= df.drop(['url', ' timedelta'], axis=1)
df= df.iloc[:,[24,25,28,39,40,58]]


# ## Pair wise scatter plots 

# In[8]:


sns.pairplot(df)
ListAttr = []
lengthOfList = len(df)
for i in df:
    print(i)
    ListAttr.append(i)
print(len(ListAttr))    


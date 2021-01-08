#!/usr/bin/env python
# coding: utf-8

# # Online News Shares

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Importing the dataset

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
df = pd.read_csv('OnlineNewsPopularity.csv')
df.head(5)


# # Encoding the dataset

# In[3]:


# No encoding required


# ## Raw data visualisation and statistics

# In[4]:


df.describe()


# In[5]:


df= df.drop(['url', ' timedelta'], axis=1)


# In[25]:


# Ploting the histograms for all the features(including the target) to visualise the data distribution
import matplotlib.pyplot as plt
df.hist(figsize=(25, 25), layout=(12,5))
plt.show()


# In[ ]:


# Ploting the histograms for all the features(including the target) to visualise the data distribution
import matplotlib.pyplot as plt
df.hist(figsize=(25, 25), layout=(12,5))
plt.show()


# # #Taking care of missing data

# In[7]:


print(df.isnull().sum()) 
print(sum(df.isnull().sum()))


# 
# ## Data cleaning Statregy:
# <br> No missing data, no out of range data<br> 
# 
# 

# ## Data visualisation (exploratory)

# In[8]:


#sns.pairplot(train_dataset[['', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
ListAttr = []
lengthOfList = len(df)
for i in df:
    print(i)
    ListAttr.append(i)
print(len(ListAttr))    


# ## Splitting the dataframe in train and test sets

# In[9]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df.iloc[:,:], test_size = 0.2, random_state = 0)
print(df_train)


# In[10]:


print(df_test)


# ## Scaling the train set features 

# In[11]:


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
df_train_scaled= min_max_scaler.fit_transform(df_train)

df_train_scaled= pd.DataFrame(data= df_train_scaled, columns=ListAttr)
print(df_train_scaled)
df_test_scaled= min_max_scaler.transform(df_test)
df_test_scaled= pd.DataFrame(data= df_test_scaled, columns=ListAttr)
print(df_test_scaled)


# ## Features' Correlation coefficients and heatmap

# In[12]:


correlation=df_train_scaled.corr(method='pearson')
print(correlation)


# In[13]:


corr_shares = correlation.iloc[:,58]


# In[14]:


corr_shares.reindex(corr_shares.abs().sort_values(ascending=False).index)


# In[19]:


plt.figure(figsize=(15,15))
df_train_corr = df_train_scaled.corr()

sns.heatmap(df_train_scaled.corr(), square=True, annot=False, cmap="Blues");


# ## Defining independent and dependent variables

# In[16]:


X_train=df_train_scaled.iloc[:, 0:-1]
y_train=df_train_scaled.iloc[:,-1]
X_test=df_test_scaled.iloc[:,0:-1]
y_test=df_test_scaled.iloc[:,-1]
print(X_train)
print(y_train)


# 
# ## Dimension reduction using PCA

# In[17]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 25)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)


# In[18]:


var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] feature
plt.xlabel('# of Features')
plt.ylabel('% variation')
plt.title('PCA Analysis')
plt.ylim(0,100.5)
plt.style.context('seaborn-whitegrid')
plt.xticks()
plt.plot(var)


#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sklearn
import numpy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[8]:


df = pd.read_csv("C:/Users/Shreya Ladhane/OneDrive/Desktop/Python p/Iris.csv")
df.tail()


# In[9]:


df = df.drop(columns=["Id"])


# In[10]:


#Display basic statistics about the data
df.describe().transpose()


# In[11]:


#checking for null values
df.isnull().sum()


# In[12]:


print('Shape of the dataset : ',df.shape)


# In[13]:


df.info()


# In[14]:


# Display the number of samples for each class
df['Species'].value_counts()


# In[15]:


#Label encoding to convert class labels into numeric form
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df['Species']


# In[16]:


df


# In[17]:


sns.countplot(x='Species',data=df,palette=['yellow','red','green'])


# In[19]:


#analysing distribution of columns values
sns.swarmplot(x=df['Species'],y=df['SepalLengthCm'],color='red')


# In[20]:


sns.swarmplot(x=df['Species'],y=df['SepalWidthCm'],color='green')


# In[21]:


sns.swarmplot(x=df['Species'],y=df['PetalLengthCm'],color='y')


# In[22]:


sns.swarmplot(x=df['Species'],y=df['PetalWidthCm'],color='orange')


# In[23]:


#Plotting the histogram of all features toghether
df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()


# In[24]:


sns.pairplot(df,hue='Species')


# In[25]:


# Compute the correlation matrix 
df.corr().transpose()


# In[ ]:





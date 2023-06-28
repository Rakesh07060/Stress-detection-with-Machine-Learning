#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[14]:


df = pd.read_csv("C:\\Users\\rakesh\\Downloads\\spam (1).csv", encoding="latin-1")

df


# In[15]:


df.shape


# In[16]:


df.head(n=10)


# In[19]:


d=np.unique(df["class"])
d


# In[21]:


d=np.unique(df["message"])
d


# In[23]:


#sparse matrix create cheyali

x=df["message"].values
y=df["class"].values
cv=CountVectorizer()
x=cv.fit_transform(x)
v=x.toarray()
print(v)


# In[24]:


first_col=df.pop("message")
df.insert(0,"message",first_col)
df


# In[25]:


#test and train chadivinav 70 percent training 30 percent testing
train_x=x[:4180]
train_y=y[:4180]

test_x=x[4180:]
test_y=y[4180:]


# In[27]:


bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)
y_pred_train=bnb.predict(train_x);
y_pred_test=bnb.predict(test_x);


# In[29]:


#score calculate chey training dhi and testing dhi
print(bnb.score(train_x,train_y)*100)
print(bnb.score(test_x,test_y)*100)


# In[30]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[31]:


from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import f1_score


# In[2]:


data = pd.read_csv('spam.csv',encoding = "latin-1")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.drop([data.columns[col] for col in range(2,5)],axis=1,inplace=True)


# In[6]:


data


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10,10))
sns.countplot(data = data, x= 'v1' )


# In[8]:


encoder = LabelEncoder()
data['v1'] = encoder.fit_transform(data["v1"])


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

vect=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)


# In[10]:


y = data['v1']


# In[11]:


data = data.drop('v1',axis = 1)


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(data, y, train_size = 0.8)


# In[13]:


X_tr_f = vect.fit_transform(X_train['v2'])
X_te_f = vect.transform(X_test['v2'])


# In[14]:


print(X_tr_f)


# In[15]:


Log = LogisticRegression()
Log.fit(X_tr_f,y_train)


# In[16]:


sv = SVC()
sv.fit(X_tr_f,y_train)


# In[17]:


nn = MLPClassifier(hidden_layer_sizes=(128,128))
nn.fit(X_tr_f,y_train)


# In[18]:


log_pre = Log.predict(X_te_f)
sv_pre = sv.predict(X_te_f)
nn_pre = nn.predict(X_te_f)


# In[19]:


print(f'accuracy of Logistic Regression : {Log.score(X_te_f,y_test)} ')
print(f'accuracy of SVM : {sv.score(X_te_f,y_test)} ')
print(f'accuracy of Neural Network : {nn.score(X_te_f,y_test)} ')


# In[20]:


print(f'F1 score of Logistic Regression : {f1_score(y_test,log_pre)} ')
print(f'F1 score of SVM : {f1_score(y_test,sv_pre)} ')
print(f'F1 score of Neural Network : {f1_score(y_test,nn_pre)} ')


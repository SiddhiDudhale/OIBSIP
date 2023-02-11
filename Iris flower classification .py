#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing library
import pandas as pd


# In[2]:


#importing dataset
df = pd.read_csv("Iris.csv")
df


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


#dropping unnecessary column
df=df.drop(columns=['Id'])
df


# In[6]:


df.Species.value_counts()


# In[7]:


df.shape


# In[8]:


#Transforming into numerical
df["Species"].replace({"Iris-setosa": 2, "Iris-versicolor": 3, "Iris-virginica": 4}, inplace = True)


# In[9]:


df.head(10)


# In[15]:


X=pd.DataFrame(df,columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]).values
y=df.Species.values.reshape(-1,1)


# In[16]:


X


# In[17]:


y


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings


# In[19]:


#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,random_state=42) 


# In[21]:


#training the model
k=7
clf=KNeighborsClassifier(k)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)


# In[22]:


metrics.accuracy_score(y_test,y_pred)*100


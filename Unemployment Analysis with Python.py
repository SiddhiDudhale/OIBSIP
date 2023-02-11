#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime as dt
import calendar


# In[2]:


#importing dataset
df =pd.read_csv("Unemployment Rate.csv")


# In[3]:


df


# In[4]:


# update column names
df.columns=["state","date","frequency","estimated unemployment rate","estimated employed","estimated labour participation rate","region", "longitude", "latitude"]


# In[5]:


#Checking for missing values
df.head()


# In[6]:


df.shape


# In[7]:


#correlation between the features of this dataset
df.corr()


# In[8]:


df.info()


# In[9]:


round(df.describe().T)


# In[10]:


# Check for Null values
df.isnull().sum()


# In[11]:


df.state.value_counts()


# In[12]:


# create a new column for month

df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['month_int'] =  df['date'].dt.month
df['month'] =  df['month_int'].apply(lambda x: calendar.month_abbr[x])
df.head()


# In[13]:


# Numeric data grouped by months

IND =  df.groupby(["month"])[['estimated unemployment rate', "estimated employed", "estimated labour participation rate"]].mean()
IND = pd.DataFrame(IND).reset_index()


# In[14]:


#Bar plot of Unemployment rate and labour participation rate
month = IND.month
unemployment_rate = IND["estimated unemployment rate"]
labour_participation_rate = IND["estimated labour participation rate"]

fig = go.Figure()

fig.add_trace(go.Bar(x = month, y = unemployment_rate, name= "Unemployment Rate"))
fig.add_trace(go.Bar(x = month, y = labour_participation_rate, name= "Labour Participation Rate"))

fig.update_layout(title="Uneployment Rate and Labour Participation Rate",
                  xaxis={"categoryorder":"array", "categoryarray":["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]})

fig.show()


# In[15]:


#Bar plot of estimated employed citizen in every month
fig = px.bar(IND, x='month',y='estimated employed', color='month',
             category_orders = {"month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]}, 
             title='estimated employed people from Jan 2020 to Oct 2020')

fig.show()


# In[16]:


# Box plot

fig = px.box(df,x='state',y='estimated unemployment rate',color='state',title='Unemployment rate')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[17]:


# bar plot unemployment rate (monthly)

fig = px.bar(df, x='state',y='estimated unemployment rate', animation_frame = 'month', color='state',
            title='Unemployment rate from Jan 2020 to Oct 2020 (State)')

fig.update_layout(xaxis={'categoryorder':'total descending'})

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]=2000

fig.show()


# In[18]:


fig = px.scatter_geo(df,'longitude', 'latitude', color="state",
                     hover_name="state", size="estimated unemployment rate",
                     animation_frame="month",scope='asia',title='Impack of lockdown on employement in India')

fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.update_geos(lataxis_range=[5,40], lonaxis_range=[65, 100],oceancolor="lightblue",
    showocean=True)

fig.show()


# In[19]:


#Regional Analysis
df.region.unique()


# In[20]:


# numeric data grouped by region

region = df.groupby(["region"])[['estimated unemployment rate', "estimated employed", "estimated labour participation rate"]].mean()
region = pd.DataFrame(region).reset_index()


# In[21]:


# scatter plot

fig = px.scatter_matrix(df, dimensions=['estimated unemployment rate','estimated employed','estimated labour participation rate'], color='region')
fig.show()


# In[22]:


#Average Unemployment Rate

fig = px.bar(region, x="region", y="estimated unemployment rate", color="region", title="Average Unemployment Rate (Region)")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[23]:


fig = px.bar(df, x='region',y='estimated unemployment rate', animation_frame = 'month', color='state',
            title='Unemployment rate from Jan 2020 to Oct 2020')

fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000

fig.show()


# In[24]:


unemployment = df.groupby(['region','state'])['estimated unemployment rate'].mean().reset_index()

unemployment.head()


# In[25]:


fig = px.sunburst(unemployment, path=['region','state'], values='estimated unemployment rate',
                  title= 'Unemployment rate in every State and Region', height=650)
fig.show()


# In[26]:


#Unemployment rate before and after Lockdown
# data p

before_lockdown = df[(df['month_int'] >= 1) & (df['month_int'] <4)]
after_lockdown = df[(df['month_int'] >= 4) & (df['month_int'] <=6)]


# In[27]:


af_lockdown = after_lockdown.groupby('state')['estimated unemployment rate'].mean().reset_index()

lockdown = before_lockdown.groupby('state')['estimated unemployment rate'].mean().reset_index()
lockdown['unemployment rate after lockdown'] = af_lockdown['estimated unemployment rate']

lockdown.columns = ['state','unemployment rate before lockdown','unemployment rate after lockdown']
lockdown.head()


# In[28]:


# Unemployment rate change after lockdown

lockdown['rate change in unemployment'] = round(lockdown['unemployment rate after lockdown'] - lockdown['unemployment rate before lockdown']
                                                /lockdown['unemployment rate before lockdown'],2)


# In[29]:


fig = px.bar(lockdown, x='state',y='rate change in unemployment',color='rate change in unemployment',
            title='Percentage change in Unemployment rate in each state after lockdown', template="ggplot2")
fig.update_layout(xaxis={'categoryorder':'total ascending'})
fig.show()


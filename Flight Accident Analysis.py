#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import warnings


# In[4]:


data = pd.read_csv('C://Users//deep//Desktop//Decodr//Case Studies_ Practice Files_ Reference Materials//Case Studies//Additional Solved Projects//Aviation accident project//Dataset1.csv')


# In[5]:


data.shape


# In[6]:


data.head(15)


# In[7]:


data.tail(15)


# In[8]:


data.isnull().sum()


# In[9]:


data['Time']=data['Time'].replace(np.nan , '00:00')


# In[10]:


data.dtypes


# In[11]:


data.dtypes


# In[12]:


data.isnull().sum()


# In[13]:


data['Time'].value_counts()


# In[14]:


data['Time'] = data['Time'].str.replace('c: ','')
data['Time'] = data['Time'].str.replace('c:','')
data['Time'] = data['Time'].str.replace('c','')
data['Time'] = data['Time'].str.replace('12\'20','12:20')
data['Time'] = data['Time'].str.replace('18.40','18:40')
data['Time'] = data['Time'].str.replace('0943','09:43')
data['Time'] = data['Time'].str.replace('22\'08','22:08')
data['Time'] = data['Time'].str.replace('114:20','00:00')


# In[15]:


data['Time'] = data['Date'] + ' ' +data['Time']

def todate(x):
    return datetime.strptime(x, '%m/%d/%Y %H:%M')

data['Time'] = data['Time'].apply(todate)


# In[16]:


data.Operator = data.Operator.str.upper()


# In[17]:


data.head()


# In[18]:


Temp = data.groupby(data.Time.dt.year)[['Date']].count()
Temp.head()


# In[19]:


Temp = Temp.rename(columns={'Date':'Count'})


# In[23]:


Temp.head(1)


# In[22]:


plt.figure(figsize=(12,6))
plt.style.use('bmh')
plt.plot(Temp.index, 'Count', data=Temp, color='blue', marker='.', linewidth=1)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by year', fontsize=15)
plt.show()


# In[24]:


import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec


gs = gridspec.GridSpec(2,2)
pl.figure(figsize=(15,10))
plt.style.use('seaborn-muted')
ax = pl.subplot(gs[0,:])
sns.barplot(data.groupby(data.Time.dt.month)[['Date']].count().index, 'Date',
            data = data.groupby(data.Time.dt.month)[['Date']].count(), color='lightskyblue', linewidth=2)
plt.xticks(data.groupby(data.Time.dt.month)[['Date']].count().index, 
           ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by month', fontsize=14)

ax = pl.subplot(gs[1,0])
sns.barplot(data.groupby(data.Time.dt.weekday)[['Date']].count().index, 'Date',
            data = data.groupby(data.Time.dt.weekday)[['Date']].count(), color='lightskyblue', linewidth=2)
plt.xticks(data.groupby(data.Time.dt.weekday)[['Date']].count().index, 
           ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by Weekday', fontsize=14)

ax = pl.subplot(gs[1,1])
sns.barplot(data[data.Time.dt.hour != 0].groupby(data.Time.dt.hour )[['Date']].count().index, 'Date',
            data = data[data.Time.dt.hour != 0].groupby(data.Time.dt.hour)[['Date']].count(), color='lightskyblue', linewidth=2)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by Hour', fontsize=14)
plt.tight_layout()
plt.show()


# In[25]:


Temp = data.copy()
Temp['isMilitary'] = Temp.Operator.str.contains('MILITARY')
Temp = Temp.groupby('isMilitary')[['isMilitary']].count()
Temp.index = ['Passenger', 'Military']
Temp


# In[26]:


Temp2 = data.copy()
Temp2['Military'] = Temp2.Operator.str.contains('MILITARY')
Temp2['Passenger'] = Temp2.Military == False
Temp2 = Temp2.loc[:,['Time', 'Military', 'Passenger']]
Temp2


# In[27]:


Temp2 = Temp2.groupby(Temp2.Time.dt.year)[['Military', 'Passenger']].aggregate(np.count_nonzero)


# In[28]:


Temp2


# In[29]:


colors = ['yellowgreen', 'lightskyblue']
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
patches, texts = plt.pie(Temp.isMilitary, colors=colors, labels=Temp.isMilitary, startangle=90)
plt.legend(patches, Temp.index, fontsize=12)
plt.axis('equal')
plt.title('Total number of accidents by flight type', fontsize=15)

plt.subplot(1,2,2)
plt.plot(Temp2.index, 'Military', data=Temp2, color='lightskyblue', marker='.', linewidth=1)
plt.plot(Temp2.index, 'Passenger', data=Temp2, color='yellowgreen', marker='.', linewidth=1)
plt.legend(fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count of accidents by flight type', fontsize=15)
plt.tight_layout()
plt.show()


# In[30]:


Fatalities = data.groupby(data.Time.dt.year).sum()
Fatalities['Proportion'] = Fatalities['Fatalities'] / Fatalities['Aboard']

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.fill_between(Fatalities.index, 'Aboard', data=Fatalities, color='skyblue', alpha=0.2)
plt.plot(Fatalities.index, 'Aboard', data=Fatalities, marker='.', color='Slateblue', alpha=0.6, linewidth=1)

plt.fill_between(Fatalities.index, 'Fatalities', data=Fatalities, color='olive', alpha=0.2)
plt.plot(Fatalities.index, 'Fatalities', data=Fatalities, marker='.', color='olive', alpha=0.6, linewidth=1)

plt.legend(fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of People', fontsize=12)
plt.title('Total number of Fatalities by Year')



plt.subplot(1,2,2)
plt.plot(Fatalities.index, 'Proportion', data=Fatalities, marker='.', color='red', linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Fatalities ratio', fontsize=12)
plt.title('Fatalities ratio by year', fontsize=15)
plt.show()


# In[31]:


Totals = pd.read_csv('C://Users//deep//Desktop//Decodr//Case Studies_ Practice Files_ Reference Materials//Case Studies//Additional Solved Projects//Aviation accident project//Dataset2.csv')


# In[32]:


Totals.head()


# In[33]:


Totals = Totals.drop(['Country Name', 'Country Code', 'Indicator Code', 'Indicator Name'], axis=1)


# In[34]:


Totals = Totals.replace(np.nan, 0)


# In[35]:


Totals = pd.DataFrame(Totals.sum())


# In[36]:


Totals.tail()


# In[37]:


Totals = Totals.drop(Totals.index[0:10])
Totals = Totals['1970':'2008']
Totals.columns = ['Sum']
Totals.index.name = 'Year'


# In[38]:


Totals.head()


# In[39]:


Fatalities = Fatalities.reset_index()


# In[40]:


Fatalities.head()


# In[41]:


Fatalities.Time = Fatalities.Time.apply(str)
Fatalities.index = Fatalities['Time']
del Fatalities['Time']
Fatalities = Fatalities['1970':'2008']
Fatalities = Fatalities[['Fatalities']]
Totals = pd.concat([Totals,Fatalities], axis=1)
Totals['Ratio'] = Totals['Fatalities'] / Totals['Sum'] * 100


# In[42]:


Totals.head()


# In[43]:


gs = gridspec.GridSpec(2,2)
pl.figure(figsize=(15,10))

ax= pl.subplot(gs[0,0])
plt.plot(Totals.index, 'Sum', data=Totals, marker='.', color='green', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Number of passengers')
plt.title('Total number of passengers by Year', fontsize=15)
plt.xticks(rotation=90)

x= pl.subplot(gs[0,1])
plt.plot(Fatalities.index, 'Fatalities', data=Totals, marker='.', color='red', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.title('Total number of Deaths by Year', fontsize=15)
plt.xticks(rotation=90)

x= pl.subplot(gs[1,:])
plt.plot(Totals.index, 'Ratio', data=Totals, marker='.', color='orange', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Ratio')
plt.title('Fatalities/Total number of passengers ratio by Year', fontsize=15)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[44]:


fig = plt.figure(figsize=(12,6))
ax1 = fig.subplots()
ax1.plot(Totals.index, 'Ratio', data=Totals, color='orange', marker='.', linewidth=1)
ax1.set_xlabel('Year', fontsize=12)
for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(45)
ax1.set_ylabel('Ratio', color='orange', fontsize=12)
ax1.tick_params('y', colors='orange')
ax2 = ax1.twinx()
ax2.plot(Fatalities.index, 'Fatalities', data=Fatalities, color='green', marker='.', linewidth=1)
ax2.set_ylabel('Number of Fatalities', color='green', fontsize=12)
ax2.tick_params('y', colors='g')
plt.title('Fatalities VS Ratio by year', fontsize=15)
plt.tight_layout()
plt.show()


# In[45]:


data.Operator = data.Operator.str.upper()
data.Operator = data.Operator.replace("A B AEROTRANSPORT", 'AB AEROTRANSPORT')

Total_by_Op = data.groupby('Operator')[['Operator']].count()
Total_by_Op = Total_by_Op.rename(columns={'Operator':'Count'})
Total_by_Op = Total_by_Op.sort_values(by='Count', ascending=False).head(15)


# In[46]:


Total_by_Op


# In[47]:


plt.figure(figsize=(12,6))
sns.barplot(y=Total_by_Op.index, x='Count', data=Total_by_Op, palette='gist_heat', orient='h')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Operator', fontsize=12)
plt.title("Total Count of the Operator", fontsize=15)
plt.show()


# In[48]:


Prop_by_Op = data.groupby('Operator')[['Fatalities']].sum()
Prop_by_Op = Prop_by_Op.rename(columns={'Operator':'Fatalities'})
Prop_by_Op = Prop_by_Op.sort_values(by='Fatalities', ascending=False)
Prop_by_OpTop = Prop_by_Op.head(15)


# In[49]:


plt.figure(figsize=(12,6))
sns.barplot(y=Prop_by_OpTop.index, x='Fatalities', data=Prop_by_OpTop, palette='gist_heat', orient='h')
plt.xlabel('Fatalities', fontsize=12)
plt.ylabel('Operator', fontsize=12)
plt.title("Total Fatalities of the Operator", fontsize=15)
plt.show()


# In[50]:


Prop_by_Op[Prop_by_Op['Fatalities'] == Prop_by_Op.Fatalities.min()].index.tolist()


# In[51]:


Aeroflot = data[data.Operator == 'AEROFLOT']
Count_by_year = Aeroflot.groupby(data.Time.dt.year)[['Date']].count()
Count_by_year = Count_by_year.rename(columns={'Date':'Count'})

plt.figure(figsize=(12,6))
plt.plot(Count_by_year.index, 'Count', data=Count_by_year, marker='.', color='red', linewidth=1)
plt.xlabel('Year', fontsize=11)
plt.ylabel('Count', fontsize=11)
plt.title('Count of accidents by year (Aeroflot)', fontsize=16)
plt.show()


# In[ ]:





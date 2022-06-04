#!/usr/bin/env python
# coding: utf-8

# In[1]:
import seaborn as sns
iris = sns.load_dataset('iris')


# In[2]:


iris


# In[3]:


iris = iris[['petal_length', 'petal_width']]


# In[4]:


iris


# In[10]:


import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.xlabel("petal length")
plt.ylabel("petal width")


# In[8]:


x = iris['petal_length']
y = iris['petal_width']


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.4, random_state=23)


# In[13]:


x_train


# In[14]:


import numpy as np
x_train = np.array(x_train).reshape(-1, 1)


# In[15]:


x_train


# In[16]:


x_test = np.array(x_test).reshape(-1,1)


# In[17]:


x_test


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


lr = LinearRegression()


# In[20]:


lr.fit(x_train, y_train)


# In[21]:


c = lr.intercept_


# In[22]:


c


# In[23]:


m = lr.coef_


# In[24]:


m


# In[25]:


y_pred_train = m*x_train + c
y_pred_train.flatten()


# In[26]:


y_pred_train1 = lr.predict(x_train)
y_pred_train1


# In[27]:


import matplotlib.pyplot as plt
plt.scatter(x_train, y_train)
plt.plot(x_train, y_pred_train1, color='red')
plt.xlabel("petal length")
plt.ylabel("petal width")


# In[28]:


y_pred_test1 = lr.predict(x_test)
y_pred_test1


# In[29]:


import matplotlib.pyplot as plt
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred_test1, color = 'red')
plt.xlabel("petal length")
plt.ylabel("petal width")


# In[30]:


pwd


# In[31]:


import pandas as pd
df = pd.read_csv('insurance.csv')
df


# In[32]:


df['sex'] = df['sex'].astype('category')
df['sex'] = df['sex'].cat.codes

df['smoker'] = df['smoker'].astype('category')
df['smoker'] = df['smoker'].cat.codes

df['region'] = df['region'].astype('category')
df['region'] = df['region'].cat.codes


# In[33]:


df


# In[34]:


df.isnull().sum()


# In[36]:


x = df.drop(columns = 'expenses')
x


# In[37]:


y = df['expenses']


# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state =23)


# In[39]:


lr_multiple = LinearRegression()
lr_multiple.fit(x_train, y_train)


# In[41]:


c = lr_multiple.intercept_
c


# In[42]:


m = lr_multiple.coef_
m


# In[44]:


y_pred_train = lr_multiple.predict(x_train)
y_pred_test = lr_multiple.predict(x_test)


# In[46]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred_test)


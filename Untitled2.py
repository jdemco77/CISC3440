#!/usr/bin/env python
# coding: utf-8

# Lab 5 
# James DeMarco
# Cisc 3410 
# monday 6:30

# In[2]:


import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


wineData = pd.read_csv('C:\DataSets\winequality-red.csv')
# this code is used to load the dataset into a variable wineData


# In[4]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wineData['quality'] = pd.cut(wineData['quality'], bins = bins, labels = group_names)

# We change the attribute of quality to take on binary values to indicate either good or bad quality.

label_quality = LabelEncoder()
wineData['quality'] = label_quality.fit_transform(wineData['quality'])

# we take the quality attribute and now change it so that 0 is bad and 1 is good
#Now our data is fit to work with


# In[5]:


wineData.head()
x = wineData.drop('quality', axis = 1)
y = wineData['quality']


# In[11]:


wineData_matrix= wineData.corr()
wineData_matrix["quality"].sort_values(ascending=False)


# In[25]:


wineData.drop("residual sugar", axis = 1, inplace = True)
wineData.drop("pH", axis = 1, inplace = True)


# In[26]:


sc = StandardScaler()
#Applying Standard scaling so we can obtain better results by bringing values to the same scale


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#Train and Test splitting of data


# In[28]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Here we apply the standard scaler to the data features


# In[29]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

#Here we run a random forest classifier which is built from decision trees.

print(classification_report(y_test, pred_rfc))
#Let's see how our model performed
#we got an accuracy of 88% which is pretty good


# In[30]:


print(confusion_matrix(y_test, pred_rfc))


# In[ ]:





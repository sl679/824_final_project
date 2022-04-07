#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Analysis Using Random Forest

# In this program, Random Forest algorithm will be used to predict cancer diagnosis. 

# ## Import the libraries needed for the analysis

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ## Dataset

# The dataset is taken from Kaggle repository - https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

# In[3]:


import pandas as pd
d1 = pd.read_csv('data.csv')


# Display the first 5 rows of the dataset.

# In[4]:


d1.head(5)


# In[5]:


d1.info()


# Drop the Unnamed column because it has all NA's

# In[6]:


d1.drop(['Unnamed: 32'], axis = 1, inplace= True)
d1.head()


# In[7]:


d1.shape


# Check missing in the dataset

# In[8]:


d1.isna().sum()


# Count the number for both diagnosis. M stands for Malignant and B for Benign. 

# In[9]:


d1['diagnosis'].value_counts()


# ## Predictor variables

# In order to see the correlation between the variables, a heatmap is plotted.

# In[10]:


plt.figure(figsize = (25,25))
sns.heatmap(d1.corr(), vmin=-1, vmax=1, annot=True)

From the heatmap, we can tell that:
- radius_worst is highly correlated with radius_mean, perimeter_mean, area_mean, concave points_mean, radius_se, area_se, perimeter_worst, area_worst, and concave points_worst; This means sense because these variables are all correlated with the size of the tumor.
- texture_mean is highly correlated with texture_worst
- smoothness_mean is highly correlated smoothness_worst
- compactness_mean is highly correlated with concavity_mean, concave points_mean, compactness_worst, concavity_worst and concave points_worst
# ## Standardizing the data

# In[11]:


std = (d1 - d1.mean())/ (d1.std())
std_data =pd.concat([d1['diagnosis'], std], axis=1)
std_data.describe()


# In[ ]:





# In[12]:


#Importing the relevant libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import xgboost


#Splitting the data

y = d1['diagnosis']
X = d1.drop(['diagnosis'], axis =1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=42)


#Creating a function to calculate accuracy/precision scores

def scores(target_test, predicted):
    ac=accuracy_score(target_test, predicted)
    precision = precision_recall_fscore_support(target_test, predicted, labels = ['M'])[0]
    recall = precision_recall_fscore_support(target_test, predicted, labels = ['M'])[1]
    fscore = precision_recall_fscore_support(target_test, predicted, labels = ['M'])[2]
    print(f" Accuracy is {ac}")
    print(f" Precision is {precision}")
    print(f" Recall is {recall}")
    print(f" F Score is {fscore}")


# In[13]:


rf = RandomForestClassifier(random_state = 42)
rf = rf.fit(X_train, y_train)
pred = rf.predict(X_test)
scores(y_test, pred)


# In[14]:


rf.feature_importances_


# In[25]:


importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]


# In[26]:


import matplotlib.pyplot as plt
 
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()


# # EDA

# In[39]:


plt.figure(figsize=(100, 100))
d1.iloc[:, 2:11].hist()
plt.rc('xtick', labelsize=0.001) 
plt.rc('ytick', labelsize=0.001) 
plt.figure(figsize=(30,30))



# In[40]:


plt.figure(figsize=(100, 100))
d1.iloc[:, 12:21].hist()
plt.rc('xtick', labelsize=0.001) 
plt.rc('ytick', labelsize=0.001) 
plt.figure(figsize=(30,30))


# In[41]:


plt.figure(figsize=(100, 100))
d1.iloc[:, 22:30].hist()
plt.rc('xtick', labelsize=0.001) 
plt.rc('ytick', labelsize=0.001) 
plt.figure(figsize=(30,30))


# In[ ]:





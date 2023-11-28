#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize']=14,8
RANDOM_SEED=42
LABELS=["Normal","Fraud"]


# In[2]:


data=pd.read_csv('creditcard.csv',sep=',')
data.head()


# In[3]:


data.info()


# In[4]:


#explortory data analysis
data.isnull().values.any()


# In[5]:


count_classes=pd.value_counts(data['Class'],sort=True)
count_classes.plot(kind='bar',rot=0)
plt.title('Transaction Class Distribution')
plt.xticks(range(2),LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[6]:


##get a fraud and normal dataset

fraud = data[data['Class']==1]

normal = data[data['Class']==0]


# In[7]:


print(fraud.shape,normal.shape)


# In[8]:


#we need to analys more amount of information from the transtion
fraud.Amount.describe()


# In[9]:


normal.Amount.describe()


# In[10]:


f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins=bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins=bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transaction')
plt.xlim((0,20000))
plt.yscale('log')
plt.show()


# In[11]:


#we will cheeck do fraudient transaction occur more often during certain tym frame? let us find out with the visaual representation
f, (ax1, ax2) = plt.subplots(2,1, sharex=True)
f.suptitle('Time of transaction vs Amount by Class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time,normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in second)')
plt.ylabel('Amount')
plt.show()


# In[12]:


#take some sample of the data
data1=data.sample(frac=0.1,random_state=1)
data1.shape


# In[13]:


data.shape


# In[14]:


#determine the number of fraud and valid transaction in the dataset
Fraud=data1[data1['Class']==1]
Valid=data1[data1['Class']==1]
outlier_fraction = len(Fraud)/float(len(Valid))


# In[15]:


print(outlier_fraction)
print("Fraud Cases : ()",format(len(Fraud)))
print("Valid Cases : ()",format(len(Valid)))


# In[16]:


#correlation
import seaborn as sns 
#get correlation of each fraction in dataset
corrmat = data1.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heatmap
g=sns.heatmap(data[top_corr_features].corr(),annot=True)


# In[17]:


#create dependent and independent adat set
columns=data1.columns.tolist()
#filter the column to remove the data set
columns = [c for c in columns if c not in ["Class"]]
#store the variable we are predicting
target="Class"
#define the random state
state=np.random.RandomState(42)
X=data1[columns]
Y=data1[target]
X_outliers = state.uniform(low=0,high=1,size=(X.shape[0],X.shape[1]))
print(X.shape)
print(Y.shape)


# In[18]:


#MODEL PREDICTION
#define the outlier detectionn method
classifiers={
    "Isolation Forest":IsolationForest(n_estimators=100,max_samples=len(X),
                                       contamination=outlier_fraction,random_state=state,verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,algorithm='auto',
                                              leaf_size=30,metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1 ,nu=0.05,
                                        max_iter=-1, )
}


# In[19]:


type (classifiers)


# In[20]:


n_outliers = len('Fraud')
for i, (clf_name, clf) in enumerate("classifiers.items()"):
#fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_prediction = clf.negative_outlier_factor_
    elif clf_name == "Support Vector Machine":
        c1f.fit(X)
        y_pred = elf.predict(X)
    else:
        clf.fit(X)
        scores_prediction = clf.decision_function(X)
        y_pred = clf.predict(X)

        
y_pred[y_pred==1]=0
y_pred[y_pred==-1]=1
n_errors = (y_pred != Y) .sum()
# Run Classification Netries
print("{}: {}".format(clf_name,n_errors))
print("Accuracy Score :")
print(accuracy_score(Y,y_pred))
print("Classification Report 1")
print(classification_report(Y,y_pred))


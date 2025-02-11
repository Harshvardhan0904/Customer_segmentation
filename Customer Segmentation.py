#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[29]:


df = pd.read_csv("C:/Users/harsh/Downloads/Mall_Customers.csv")
df.drop(columns=['CustomerID','Age'], axis =1 ,inplace =True)


# In[36]:


df.rename(columns={'Annual Income (k$)':'income','Spending Score (1-100)':'spending'},inplace=True)
df.head()


# In[37]:


sc = StandardScaler()
features = ['income', 'spending']
df_scaled = sc.fit_transform(df[features])


# In[48]:


inertia = []
k_val = range(1,11)
for i in k_val:
    k = KMeans(n_clusters=i , random_state=5)
    k.fit(df_scaled)
    inertia.append(k.inertia_)


# In[50]:


plt.figure(figsize=(10,5))
plt.plot(k_val,inertia , marker='o' , linestyle = '--')
plt.title('ELBOW METHOD')
plt.xlabel('clusters')
plt.ylabel('inertia')


# In[69]:


optimal = 5
kmean = KMeans(n_clusters=optimal,random_state=10)
df['cluster']  = kmean.fit_predict(df_scaled)


# In[76]:


plt.figure(figsize=(8, 5))
for cluster in range(optimal):
    cluster_data = df[df['cluster'] == cluster]
    plt.scatter(cluster_data['income'], cluster_data['spending'], label=f'Cluster {cluster}')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Avg Spending')
plt.legend()
plt.show()


# In[ ]:





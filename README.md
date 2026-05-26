# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize the Dataset Load the customer dataset and select the required features such as Annual Income and Spending Score.

2.Choose the Number of Clusters (K) Decide the number of clusters to form using methods like the Elbow Method.

3.Assign Data Points to Clusters Calculate the distance between each data point and cluster centroid, then assign each point to the nearest centroid.

4.Update Cluster Centroids Recalculate the centroid of each cluster by taking the mean of all data points assigned to it.

5.Repeat Until Convergence Continue assigning points and updating centroids until the centroids no longer change significantly and final customer segments are formed.

## Program:
/*
Program to implement the K Means Clustering for Customer Segmentation.

Developed by: Jones Benedict A P

RegisterNumber:212224040142 
*/
```

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []

for i in range(1, 11):

    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=0,
        n_init=10
    )

    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')

plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

km = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=0,
    n_init=10
)

km.fit(data.iloc[:, 3:])

y_pred = km.predict(data.iloc[:, 3:])

print(y_pred)

data["cluster"] = y_pred

df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

plt.figure(figsize=(10,6))

plt.scatter(
    df0["Annual Income (k$)"],
    df0["Spending Score (1-100)"],
    c="red",
    label="cluster0"
)

plt.scatter(
    df1["Annual Income (k$)"],
    df1["Spending Score (1-100)"],
    c="black",
    label="cluster1"
)

plt.scatter(
    df2["Annual Income (k$)"],
    df2["Spending Score (1-100)"],
    c="blue",
    label="cluster2"
)

plt.scatter(
    df3["Annual Income (k$)"],
    df3["Spending Score (1-100)"],
    c="green",
    label="cluster3"
)

plt.scatter(
    df4["Annual Income (k$)"],
    df4["Spending Score (1-100)"],
    c="magenta",
    label="cluster4"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")

plt.legend()
plt.show()
```



## Output:

<img width="385" height="659" alt="image" src="https://github.com/user-attachments/assets/f1b9cfcd-8a9c-4fd2-9d05-b0c3633bb29a" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

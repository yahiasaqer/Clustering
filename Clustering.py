import pandas as pd

iris = pd.read_csv('IRIS.csv')
# print(iris.head(5))

# split data attribute and label attribute
attributes = iris.drop(['species'], axis=1)
labels = iris['species']

#Import and create KMeans object
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3) #K value is 3 here, so we have three centroids.
model.fit(attributes) #we are dealing with the attributes only now.

# find/predict the clusters list for the dataset
y_pred = model.predict(attributes)
print(y_pred) #the result is a list of cluster numbers (0, 1, 2), if n_clusters=4 the cluster numbers are: (0, 1, 2, 3) and so on.


# evaluation stage
from sklearn import metrics
contingecyMatric = metrics.cluster.contingency_matrix(labels, y_pred)
print(contingecyMatric)

ari = metrics.cluster.adjusted_rand_score(labels, y_pred) #ari is adjusted rand index
print('ari', ari)
nmi = metrics.cluster.normalized_mutual_info_score(labels, y_pred)
print('nmi', nmi)
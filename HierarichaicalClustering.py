import pandas as pd

iris = pd.read_csv('IRIS.csv')
# print(iris.head(5))

# split data attributes and label attribute
attributes = iris.drop(['species'], axis=1)
labels = iris['species']

# import and build hieratchy cluster model
from scipy.cluster.hierarchy import linkage, dendrogram
hc = linkage(attributes, 'single') 
# print(hc) #the result is a list of the closest pairs, the distance  between each pair, and the cluster that pair belongs to.

# plot the dendogram
samplelist = range(1, 151)# make a list for the data samples

# import pylot libray
from matplotlib import pyplot as plt

plt.figure(figsize=(30, 15))
dendrogram(hc,
           orientation='top',
           labels=samplelist,
           distance_sort='descending',
           show_leaf_counts='true')

plt.show()
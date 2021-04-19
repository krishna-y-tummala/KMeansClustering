#Change directory if necessary
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\KMeans')
os.getcwd()

#IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#IMPORT DATA
df = pd.read_csv('C:\\Users\\User\\Documents\\College_Data',index_col=0)

#EDA
print('\n',df.head(),'\n')

print('\n',df.info(),'\n')

print('\n',df.describe(),'\n')

#Grad.Rate vs Room.Board ScatterPlot
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='Room.Board',y='Grad.Rate',hue='Private',palette='coolwarm')
plt.savefig('Grad.Rate vs Room.Board ScatterPlot.jpg')
plt.show()

#Outstate vs F.Undergrad Scatter
plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='Outstate',y='F.Undergrad',hue='Private',palette='coolwarm')
plt.savefig('Outstate vs F.Undergrad Scatter.jpg')
plt.show()

#Outstate Tuition Histogram
g = sns.FacetGrid(data=df,hue='Private',sharey=True,height=6,aspect=2,palette='coolwarm')
g = g.map(plt.hist,'Outstate',bins=30,alpha=0.6)
g.savefig('Outstate Tuition Histogram.jpg')
plt.show()

#Grad.Rate Histogram
g = sns.FacetGrid(data=df,hue='Private',sharey=True,height=6,aspect=2,palette='coolwarm')
g = g.map(plt.hist,'Grad.Rate',bins=25,alpha=0.6)
g.savefig('Graduation Rate Histogram')
plt.show()

#One College has Graduation Rate above 100%, we need to make it 100%
df.loc['Cazenovia College','Grad.Rate'] = 100

#Corrected Grad Rate Histogram
g = sns.FacetGrid(data=df,hue='Private',sharey=True,height=6,aspect=2,palette='coolwarm')
g = g.map(plt.hist,'Grad.Rate',bins=25,alpha=0.6)
g.savefig('Corrected Grad Rate Histogram')
plt.show()

#KMeans Algorithm
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
#We know number of cluster is 2 because there are only Public and Private Schools
#A better way to choose the number of clusters is to plot the SSE vs. K and choose the best K

X= df.drop('Private',axis=1)
y = df['Private']

kmeans.fit(X)
#We dont split data because it is an unsupervised algorithm
print('Kmeans cluster centers:','\n')
kmeans.cluster_centers_

#Normally we cannot evaluate an Unsupervised job, but in this case we have the Private column to compare
#We convert the Private column into a categorical column, 1 if Private 0 if not Private
df['Cluster'] = df['Private'].apply(lambda x: 1 if x=='Yes' else 0)

print('\n',df.tail(),'\n')

#METRICS
from sklearn.metrics import classification_report,confusion_matrix,plot_confusion_matrix

print('Confusion Matrix:\n',confusion_matrix(df['Cluster'],kmeans.labels_))
print('Classification Report:\n',classification_report(df['Cluster'],kmeans.labels_))

#END
#THE Metrics may vary beacause the algorithm does not recognize the labels correctly sometimes
#The confusion matrix may get inverted sometimes. It is the trade off for using metrics on an Unsupervised Algorithm

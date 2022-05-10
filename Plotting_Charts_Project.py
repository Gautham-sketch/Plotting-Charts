from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import csv

data = []
with open("Final_Prjoect.csv",'r') as f:
  reader = csv.reader(f)
  for row in reader:
    data.append(row)

radiuses = []
masses = []
gravity_data = []
X = []
for index,mass in enumerate(masses):
  temp = [radiuses[index],mass]
  X.append(temp)

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init="k-means++",random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)

plt.Figure(figsize = (10,5))
sns.lineplot(range(1,11),wcss,marker = 'o',color = 'red')
plt.title("Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

Y = []
for index,mass in enumerate(masses):
  temp = [radiuses[index],mass]
  Y.append(temp)

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init="k-means++",random_state=42)
  kmeans.fit(Y)
  wcss.append(kmeans.inertia_)

plt.Figure(figsize = (10,5))
sns.lineplot(range(1,11),wcss,marker = 'o',color = 'blue')
plt.title("Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()
"""
All Rights Reserved
@author Morteza Emadi
this code will use the features of the 39 homes and the weighting values corresponding
to them for 2-level clustering. Note: In contrast to the 2nd level clustering in the
first level we know that we need two categories of cold & warm climate homes.(K value is known)
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

### this is the range within which I compare num of Clusters by implementing silhoute  index and so on...
clusterRange = range(2, 10)
df = pd.read_excel('data\clustering\GRA.xlsx', index_col=None, header=None).to_numpy()
normalized = pd.read_excel('encoded_labeled_ver1.xlsx', index_col=0)
coeff = pd.read_excel('Mutual_info_GRADES.xlsx', index_col=0)
coeff = coeff['avg_score']
###names of main feattures 29 ones
colNames = coeff.index.to_numpy()
allData = []
data = []
c = []

for i in coeff.index:
    """"making all data with multiplying features into their MUTUAL  INFO score """
    allData.append(np.array(normalized[i].values))
    if coeff[
        i] != 0:
        data.append(coeff[i] * allData[-1])
#by target I meant the referene feature (elec1_cold_term)
target = normalized.to_numpy()[:, 0]
allData = np.array(allData).T
data = np.array(data).T
coeff = coeff.values


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)]) * 10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i] - cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i] - ci[j])

    return np.max(values)


def dunn(data, label):

    k_list = []
    for l in np.unique(label):
        k_list.append(data[label == l])

    deltas = np.ones([len(k_list), len(k_list)]) * 1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas) / np.max(big_deltas)
    return di


def Get_Col(name):
    idx = np.unravel_index(np.argmax(df == name), df.shape)
    outCol = df[:, idx[1]]
    return outCol, idx

def Return_Table(name):
    _, idx = Get_Col(name)

    startRow = idx[0] + 1
    startCol = idx[1]

    r = startRow
    c = startCol

    while r < df.shape[0] and c < df.shape[1]:
        if str(df[r, c]) != 'nan':
            r += 1
        else:
            break
    endRow = r - 1

    r = startRow
    while r < df.shape[0] and c < df.shape[1]:
        if str(df[r, c]) != 'nan':
            c += 1
        else:
            break
    endCol = c - 1

    return df[startRow:endRow + 1, startCol:endCol + 1]


def Scatter_2D(colNames, col1, col2, labels, alg, sctdata=allData, scttarget=target):
    idx1 = colNames == col1
    idx2 = colNames == col2

    if np.sum(idx1) == 0:
        data1 = scttarget
    else:
        data1 = sctdata[:, idx1]

    if np.sum(idx2) == 0:
        data2 = scttarget
    else:
        data2 = sctdata[:, idx2]

    plt.figure()
    for j, i in enumerate(np.unique(labels)):
        ii = (labels == i)

        d1 = data1[ii]
        d2 = data2[ii]

        plt.scatter(d1, d2, label='cluster {}'.format(j + 1))

    plt.legend()
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('{} Algorithm \n {} vs {} '.format(alg, col1, col2))
    plt.savefig('{}-{}.png'.format(col1, col2))
    plt.show()


def Cluster(algorithm, data, **kwargs):
    model = algorithm(**kwargs)
    model.fit(data)

    try:
        label = model.predict(data)
    except:
        label = model.fit_predict(data)
        pass

    return model, label


homeId = Return_Table('homeid')

"""we wanted 2 clustering, so here we use the all data and corresponding mutual info grades to make 1st and 2nd clustering data prepared"""
cluster1Data = []
for i in colNames:
    if (i in ['humidity1_cold_term', 'realfeel1_cold_term', 'temperature1_cold_term', 'windspeed1_cold_term']):
        idx = np.argmax(colNames == i)
        cluster1Data.append(allData[:, idx] * coeff[idx])

cluster1Data = np.array(cluster1Data).T

# -------------------------- data aggregating for Clustering 2--------------------------------------
cluster2Data = []
for i in colNames:
    idx = np.argmax(colNames == i)
    if (coeff[idx] != 0) & (
    not (i in ['humidity1_cold_term', 'realfeel1_cold_term', 'temperature1_cold_term', 'windspeed1_cold_term'])):
        cluster2Data.append(allData[:, idx] * coeff[idx])

deleting = np.array([])

for i in colNames:
    idx = np.argmax(colNames == i)
    if (coeff[idx] == 0) | (
    (i in ['humidity1_cold_term', 'realfeel1_cold_term', 'temperature1_cold_term', 'windspeed1_cold_term'])):
        deleting = np.append(deleting, idx)
colNames2 = np.delete(colNames, deleting.astype(int), 0)

cluster2Data = np.array(cluster2Data).T

from sklearn.metrics import silhouette_score

# %% K means
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

SS = []
DUN = []
for clusters in clusterRange:
    model, label = Cluster(KMeans, cluster1Data, n_clusters=clusters, random_state=1)

    SS.append(silhouette_score(cluster1Data, label))
    DUN.append(dunn(cluster1Data, label))

plt.figure()
plt.plot(clusterRange, SS)
plt.title('silhouette_score for K means')
plt.grid()
plt.xticks(clusterRange)
plt.xlabel('K')
plt.ylabel('silhouette score')
plt.savefig('silhouette_score_Kmeans.png')
plt.show()

plt.figure()
plt.plot(clusterRange, DUN)
plt.title('dunn score for K means')
plt.grid()
plt.xticks(clusterRange)
plt.xlabel('K')
plt.ylabel('dunn score')
plt.savefig('dunn_score_Kmeans.png')
plt.show()
# plt.close()

print('\nK means :')
print('\tdunn_score : ', max(DUN))
print('\tsilhouette_score: ', max(SS))

# %% AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering

SS = []
DUN = []
for clusters in clusterRange:
    model, label = Cluster(AgglomerativeClustering, cluster1Data, n_clusters=clusters)

    SS.append(silhouette_score(cluster1Data, label))
    DUN.append(dunn(cluster1Data, label))

plt.figure()
plt.plot(clusterRange, SS)
plt.title('silhouette_score for AgglomerativeClustering')
plt.grid()
plt.xticks(clusterRange)
plt.xlabel('K')
plt.ylabel('silhouette score')
plt.savefig('silhouette_score_AgglomerativeClustering.png')
plt.show()

plt.figure()
plt.plot(clusterRange, DUN)
plt.title('dunn score for AgglomerativeClustering')
plt.grid()
plt.xticks(clusterRange)
plt.xlabel('K')
plt.ylabel('dunn score')
plt.savefig('dunn_score_AgglomerativeClustering.png')
plt.show()

print('\n\n\n\n\nAgglomerativeClustering :')
print('\tdunn_score : ', max(DUN))
print('\tsilhouette_score: ', max(SS))

from sklearn.mixture import GaussianMixture

model, label = Cluster(GaussianMixture, data, n_components=data.shape[1])

print('\n\n\n\n\nGMM :')
print('\tdunn_score : ', dunn(data, label))
print('\tsilhouette_score: ', silhouette_score(data, label))

########################################################
"""We know for the 1st clustering I want K=2, so here I am scatter plotting to analyse visulaly what 1st clustring
have done, we use the reference feature with one of the most influencial feature"""

modelk, labelk = Cluster(KMeans, cluster1Data, n_clusters=2, random_state=1)
modelag, labelag = Cluster(AgglomerativeClustering, cluster1Data, n_clusters=2)
# model,label = Cluster(GaussianMixture,data,n_components=data.shape[1])
modelsh, labelsh = Cluster(MeanShift, cluster1Data, bandwidth=estimate_bandwidth(cluster1Data, quantile=0.85),
                           bin_seeding=False)

###Running Scatter Func
Scatter_2D(colNames, 'elec1_cold_term', 'occupy_period', labelk, "KMeans")
Scatter_2D(colNames, 'elec1_cold_term', 'occupy_period', labelsh, "MeanShift")
print("silhouette for KMeans,k=2", silhouette_score(cluster1Data, labelk))
print("silhouette for meanshift,k=2", silhouette_score(cluster1Data, labelsh))
print("silhouette for Agglomerative Clustering,k=2", silhouette_score(cluster1Data, labelag))
print("You have to run dbscan tooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo!!!!!!!!!!")


def print_by_id(label, index=None):
    """This will tell which home is corresponding to which cluster exaclty by their homeid"""
    if index == None:
        index = range(len(label) - 1)

    for l in np.unique(label):
        print('\nfor label = ', l)
        for i, j in zip(index, label):
            if j == l:
                print('\t', homeId[i, 0])


firstModel, label4id = Cluster(KMeans, cluster1Data, n_clusters=2, random_state=1)
print_by_id(label4id)

########################################################
# %% multistep clustering

firstModel, label = Cluster(KMeans, cluster1Data, n_clusters=2, random_state=1)
# firstModel,label = Cluster(AgglomerativeClustering,cluster1Data,n_clusters=2)
# firstModel,label = Cluster(GaussianMixture,cluster1Data,n_components=data.shape[1])

"""By using the labels from the first step(above) we start 2nd step of clustering with the
remaining features(29-4=25featues)"""
# second step

for l in np.unique(label):
    idx = (label == l)
    d = cluster2Data[idx]
    if len(d) > 2:

        secondRange = range(2, len(d))

        SS = []
        DUN = []
        for clusters in secondRange:
            # model = KMeans(n_clusters=clusters, random_state=1)
            # model.fit(data)
            # label = model.predict(data)
            print(d.shape)
            m, ll = Cluster(KMeans, d, n_clusters=clusters, random_state=1)
            # model,label = Cluster(KMeans,cluster1Data,n_clusters=clusters, random_state=1)

            SS.append(silhouette_score(d, ll))
            DUN.append(dunn(d, ll))
        plt.figure()
        plt.plot(secondRange, SS)
        plt.title('2nd clustering in 1st cluster of K1 \n Silhouette Score by K-means')
        plt.grid()
        plt.xticks(secondRange)
        plt.xlabel('K2')
        plt.ylabel('silhouette score')
        plt.savefig('silhouette_score_Kmeans.png')
        plt.show()
        # plt.close()

        plt.figure()
        plt.plot(secondRange, DUN)
        plt.title('dunn score for K means')
        plt.grid()
        plt.xticks(secondRange)
        plt.xlabel('K')
        plt.ylabel('dunn score')
        plt.savefig('dunn_score_Kmeans.png')
        plt.show()


"""preparing data and RESULTS of 2nd clustering to get scatters of their clusters"""

idx = (label == 0)
d = cluster2Data[idx]
m, lll = Cluster(KMeans, d, n_clusters=5, random_state=1)
idx2 = (label == lll)
d2 = cluster2Data[idx]
target2 = target[idx]

Scatter_2D(colNames2, 'elec1_cold_term', 'roomarea', lll, "2nd clustering in 1st k1", d2, target2)
barlist = []
##############################################################
secondModels = {}
for l in np.unique(label):
    idx = (label == l)
    d = cluster2Data[idx]
    if len(d) > 1:
        m, ll = Cluster(KMeans, d, n_clusters=5, random_state=1)
        print_by_id(ll)
        secondModels[l] = {'model': m, 'label': ll}
    else:
        print_by_id(ll)
        secondModels[l] = {'model': firstModel, 'label': ll}
print(secondModels)


from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy
import numpy as np
import os 
import matplotlib.pyplot as plt
import heapq
from matplotlib.font_manager import FontProperties

def dist_to_similarity(dist):
    row_mean = np.matmul(dist,np.ones(dist.shape))/dist.shape[0]
    col_mean =  np.matmul(np.ones(dist.shape), dist)/dist.shape[1]
    return -1/2 * (dist - row_mean - col_mean + np.mean(dist))

def mds(S, new_dim):
    eigen_values, eigen_vectors = np.linalg.eig(S)
    diag = np.diag(eigen_values)**0.5
    X = np.matmul(np.eye(new_dim, S.shape[0]), diag)
    X = np.matmul(X, eigen_vectors.T)
    return X.T

def floyd_warshall(dist):
    N = dist.shape[0]
    new_dist = np.array(dist)
    test = np.array(new_dist)
    for k in range(N):
        for i in range(N):
            for j in range(N):
                new_dist[i][j] = min(new_dist[i][j], new_dist[i][k] + new_dist[k][j])
    #print(np.sum(test - new_dist))
    return new_dist

def isomap(data, dist, new_dim, neighbors):
    N = dist.shape[0]
    nn = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            heapq.heappush(nn[i], (dist[i][j], j))
    #nbrs = NearestNeighbors(n_neighbors = neighbors + 1, algorithm='brute').fit(data)
    new_dist = np.ones(dist.shape)*1000000000
    #distance, indices = nbrs.kneighbors(data)
    for i in range(N):
        for _ in range(neighbors):
            best = heapq.heappop(nn[i])
            new_dist[i][best[1]] = new_dist[best[1]][i] = best[0]
            #new_dist = csr_matrix(new_dist)
    #new_dist = scipy.sparse.csgraph.floyd_warshall(csgraph = new_dist)
    new_dist = floyd_warshall(new_dist)
    return mds(dist_to_similarity(new_dist), new_dim)


def read_data(file_name):        
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output = []
    with open(dir_path + "/" + file_name, 'r') as zoo:
        for line in zoo:
            output.append(line.split(','))
    return output

def clean_input(inp):
    new_input = []
    nummeric = []
    group = []
    for i in range(len(inp)):
        new_input.append(inp[i][0])
        nummeric.append(inp[i][1:13])
        one_hot_enc = [0] * 7
        one_hot_enc[int(inp[i][13]) // 2] = 1
        nummeric[i].extend(one_hot_enc)
        nummeric[i].extend(inp[i][14:17])
        nummeric[i] = [int(j) for j in nummeric[i]]
        group.append(int(inp[i][-1]))
    nummeric = np.array(nummeric)
    nummeric = nummeric - np.mean(nummeric, axis=0)
    #nummeric = nummeric*np.std(nummeric, axis=0)
    #for i in range(len(new_input)):
    #    new_input[i].extend(nummeric[i])
    return np.array(new_input), nummeric, np.array(group)

def euclidean(x):
    N = x.shape[0]
    distance = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            distance[i][j] = distance[j][i] = np.linalg.norm(x[i] - x[j]) ** 2
    return distance

def weighted_dist(x):
    N = x.shape[0]
    distance = np.zeros((N, N))
    scale = np.std(x, axis=0)
    tot = np.sum(scale)
    for i in range(N):
        for j in range(i + 1, N):
            dist = 0
            for k in range(x.shape[1]):
                dist = dist + scale[k]*(x[i, k] - x[j, k])**2
            dist = (dist / tot)**0.5 
            distance[i][j] = distance[j][i] = dist
    return distance


#Testing that similarity matrix returns as expected.
test = np.array([[0, 2, 3], [2, 0, 6], [3, 6, 0]])
similarity = dist_to_similarity(test)
u, v = np.linalg.eig(test)
y = np.arange(3)
#print(np.matmul(test, y))
def td(u, v, x):
    s = np.matmul(v.T, x)
    s = np.matmul(np.diag(u), s)
    return np.matmul(v, s)
#print(td(u, v, y))


label, data, group = clean_input(read_data('zoo.data'))
dist = euclidean(data)

#PCA
pca = PCA(n_components = 2)
y_pca = pca.fit_transform(data)

mds_dist = weighted_dist(data)

#MDS
similarity = dist_to_similarity(mds_dist)
y_mds = mds(similarity, 2)

print(np.sum(y_pca - y_mds))
#ISOMAP
y_iso = isomap(data, dist, 2, 4)

using = "mds"

fig, ax = plt.subplots()
for k in range(1,8):
    if using == "pca":
        y = y_pca
    elif using == "mds":
        y = y_mds
    elif using == "iso":
        y = y_iso
    y_tmp = y[group == k]
    ax.scatter(y_tmp[:,0], y_tmp[:,1])

fontP = FontProperties()
fontP.set_size('xx-small')
ax.legend([str(i) for i in range(1, 8)], title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.subplots_adjust(right = 0.8)
plt.title(using.upper() + ": For groups 1-7")
plt.savefig(using + ".png")
#for i,name in enumerate(group):
 #   ax.annotate(str(name), (y[i,0], y[i, 1]))
plt.show()
import support_functions as sup_funcs
import random
from datetime import datetime
import numpy as np

random.seed(datetime.now().timestamp())
N = 1000
D = 2
K = 3

X = sup_funcs.generate_points(N)

distance_k_means, amount_of_clusters_k_means = sup_funcs.plot_k_means(X, K, N)
print("Amount of clusters in K-mean algorithm: ", amount_of_clusters_k_means)
print("Mean of a distance in K-mean algorithm: ", distance_k_means)

distance_dbscan, amount_of_clusters_dbscan = sup_funcs.dbscan(np.asarray(X))
print("Amount of clusters in DBSCAN algorithm: ", amount_of_clusters_dbscan)
print("Mean of a distance in DBSCAN algorithm: ", distance_dbscan)


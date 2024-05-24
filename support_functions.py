import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons


def generate_points(n):
    points = []

    for i in range(n):
        point = []

        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        point.append(x)
        point.append(y)
        points.append(point)
    return points


def distance(point1, point2):
    return math.dist(point1, point2)


def k_means(x, k, n, max_iters=100):
    centroid_indices = np.random.choice(n, k, replace=False)

    centroids = []

    for i in range(k):
        centroids.append(x[centroid_indices[i]])

    avg_distances = []
    labels = []

    for _ in range(max_iters):

        distances = [[] for _ in range(k)]

        for i in range(k):
            for j in range(n):
                distances[i].append(distance(x[j], centroids[i]))

        labels = np.argmin(distances, axis=0)

        for i in range(k):

            centroid_x = []

            for j in range(n):
                if labels[j] == i:
                    centroid_x.append(x[j])

            centroids[i] = np.mean(centroid_x, axis=0)

            avg_distances = [np.mean(distances[k]) for k in range(k)]

    return centroids, avg_distances, labels

# об’єднання точок на кластери за відстанню між ними
def plot_k_means(x, k, n):
    centroids, avg_distances, labels = k_means(x, k, n)

    first_coords = [sub_x[0] for sub_x in x]
    second_coords = [sub_x[1] for sub_x in x]

    plt.scatter(first_coords, second_coords, c=labels, cmap='viridis')

    first_coords = [centroid[0] for centroid in centroids]
    second_coords = [centroid[1] for centroid in centroids]

    plt.scatter(first_coords, second_coords, marker='x', c='red', s=100)
    plt.title('K-means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Cluster')
    plt.show()

    return np.mean(avg_distances), len(np.unique(labels))

def dbscan(x, eps=.04):
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(x)

    plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis')
    plt.title("DBSCAN Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    clusters = []

    for i in np.unique(labels):
        clusters.append(x[labels == i])

    avg_distances = []

    for cluster in clusters:
        cluster_dot = cluster.tolist()
        if len(cluster_dot) > 1:
            cluster_distance = []
            for i in range(len(cluster_dot)):
                for j in range(i + 1, len(cluster_dot)):
                    cluster_distance.append(distance(cluster_dot[i], cluster_dot[j]))
            avg_distances.append(np.mean(cluster_distance))

    return np.mean(avg_distances), len(np.unique(labels))

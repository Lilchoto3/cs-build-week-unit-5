import numpy as np
from scipy.spatial import distance

class KMeans:

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.labels_ = None
        self.centroids_ = None

    def fit(self, data):
        '''
        Fits n_clusters centroids to the data and clusters them.
        data cannot be empty and must have more points than
        n_clusters.

        input
        -----
        data
            2D numpy array (convert DataFrames via df.to_numpy())
        '''
        # initialize centroids
        sample_idx = [(len(data) / self.n_clusters) * x for x in range(self.n_clusters)]
        self.centroids_ = [data[x] for x in sample_idx]

        iter_count = 0
        while iter_count < self.max_iter:
            prev_labels = self.labels_
            # calculate centroid labels for each point
            dists = distance.cdist(data, self.centroids_, 'euclidean')
            self.labels_ = np.argmin(dists, axis=1)

            # stop early if the new labels are the same.
            if self.labels_ == prev_labels:
                break

            # calculate new centroids
            pointlist = [[] for x in self.centroids_]
            self.centroids_ = []
            for i in range(len(data)):
                pointlist[self.labels_[i]].append(data[i])
            for k in range(self.n_clusters):
                self.centroids_[k] = np.mean(pointlist[k], axis=1)
            
            iter_count += 1
        
        def predict(self, data):
            '''
            Takes data and assigns clusters to the points based on
            current centroids stored in centroids_

            input
            -----
            data
                2D numpy array (convert DataFrames via df.to_numpy())

            output
            ------
            labels
                ndarray of the labels each point is closest to
            '''
            dists = distance.cdist(data, self.centroids_, 'euclidean')
            labels = np.argmin(dists, axis=1)
            return labels

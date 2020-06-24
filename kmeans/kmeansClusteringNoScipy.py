import math
import statistics
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
        sample_idx = [x for x in range(self.n_clusters)]
        # print(sample_idx)
        self.centroids_ = [data[x] for x in sample_idx]

        iter_count = 0
        while iter_count < self.max_iter:
            print (f'Iteration: {iter_count}')
            prev_labels = self.labels_
            # calculate centroid labels for each point
            # TODO: calculate n-dimensional euclidean distance
            # and assign labels based on closest centroid
            self.labels_ = []
            for row in data:
                dists = []
                for c in self.centroids_:
                    dist = float(0)
                    # get absolute value of distance
                    # on each dimension
                    for i in range(len(row)):
                        dist += abs(row[i]-c[i]) ** 2
                    dist = math.sqrt(dist)
                    dists.append(dist)
                
                # go through each distance and get the smallest
                for i in range(self.n_clusters):
                    if dists[i] == min(dists):
                        self.labels_.append(i)

            # stop early if the new labels are the same.
            if self.labels_ == prev_labels:
                break

            # calculate new centroids
            pointlist = [[] for x in self.centroids_]
            self.centroids_ = []
            for i in range(len(data)):
                pointlist[self.labels_[i]].append(data[i])
            for k in range(self.n_clusters):
                means = [0 for x in range(len(pointlist[k][0]))]
                # add up distances for each dim
                for row in pointlist[k]:
                    for n in range(len(row)):
                        means[n] += row[n]
                # divide by length of pointlist for each dim
                for n in range(len(pointlist[k][0])):
                    means[n] = means[n] / len(pointlist[k])
                # append new centroid
                self.centroids_.append(means)
                

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
        labels = []
        for row in data:
            dists = []
            for c in self.centroids_:
                dist = float(0)
                # get absolute value of distance
                # on each dimension
                for i in range(len(row)):
                    dist += abs(row[i]-c[i]) ** 2
                dist = math.sqrt(dist)
                dists.append(dist)
            
            # go through each distance and get the smallest
            for i in range(self.n_clusters):
                if dists[i] == min(dists):
                    labels.append(i)
        return labels

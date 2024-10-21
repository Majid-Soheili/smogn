
from sklearn_extra.cluster import KMedoids

import pandas as pd
import numpy as np

from smogn.Sampler import Sampler


class UnderSampler(Sampler):
    def __init__(self, data, index, percentage, method="random", seed=None):

        """
        :param data (pd.DataFrame):
        :param index:
        :param percentage: it should be between 0 and 1
        :param method:  it should be either "random", "cluster" or "density"
        :param seed:  it should be an integer and used for reproducibility
        """
        self.method = method
        super().__init__(data, index, percentage, seed)
    def provide_under_sampled_data(self):

        if self.method == "random":
            self._random_sampling()
        elif self.method == "cluster":
            self._cluster_sampling()
        elif self.method == "density":
            self.density_based_undersample()
        else:
            raise ValueError("Invalid method")

        return self._new_data

    def _random_sampling(self):
        # Randomly sample the datas
        self._new_data = self._original_data.sample(n=self.num_new_data, replace=False).copy(deep=True)
        self._new_data.reset_index(drop=True, inplace=True)

    def _cluster_sampling(self):
        # Maximum number of clusters
        num_clusters = self.num_new_data
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.seed)
        self._original_data['cluster'] = kmeans.fit_predict(self._original_data)

        # Get the cluster centroids
        centroids = kmeans.cluster_centers_
        self.new_data = pd.DataFrame(centroids, columns=self._original_data.columns)

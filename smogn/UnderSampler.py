
from sklearn.cluster import KMeans
import pandas as pd
class UnderSampler:
    def __init__(self, data, index, percentage, seed=None):

        self._original_data = data.iloc[index, :].copy(deep=True)
        self._original_data.reset_index(drop=True, inplace=True)
        self.percentage = percentage
        self.num_new_data = int(self.percentage * len(self._original_data))
        self.new_data = None
        self.seed = seed

    def provide_under_sampled_data(self):
        self._random_sampling()
        #if self.percentage > 0.9:
        #     self._random_sampling()
        #else:
        #    self._cluster_sampling()
        return self.new_data

    def _random_sampling(self):
        # Randomly sample the data
        self.new_data = self._original_data.sample(n=self.num_new_data, replace=False)
        self.new_data.reset_index(drop=True, inplace=True)

    def _cluster_sampling(self):
        # Maximum number of clusters
        num_clusters = self.num_new_data
        kmeans = KMeans(n_clusters=num_clusters, random_state=self.seed)
        self._original_data['cluster'] = kmeans.fit_predict(self._original_data)

        # Get the cluster centroids
        centroids = kmeans.cluster_centers_
        self.new_data = pd.DataFrame(centroids, columns=self._original_data.columns)

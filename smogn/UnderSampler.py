from sklearn.cluster import DBSCAN
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
        super().__init__(data, index, percentage, seed=seed)
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

        self._compute_distance_matrix()
        self._normalize_distance_matrix()

        dbscan = DBSCAN(eps=0.2, min_samples=self.nk, metric='precomputed')
        dbscan.fit(self._distance_matrix)

        df = self._original_data.copy(deep=True)
        df['cluster'] = dbscan.labels_

        # Filter out noise points (cluster label -1)
        df = df[df['cluster'] != -1]

        if len(df) == 0 or self.num_new_data > len(df):
            self._logger.warning("The outliers are too much, consider changing epsilon or min_samples of DBSCAN.")
            self._logger.warning("Returning random sample.")
            return self._random_sampling()

        self._new_data = pd.DataFrame()
        unique_clusters = df['cluster'].unique()
        sample_rate = (self.num_new_data / len(df))
        for cluster in unique_clusters:
            cluster_data = df[df['cluster'] == cluster]
            sample_size = int(round(len(cluster_data) * sample_rate))
            if len(cluster_data) > sample_size:
                cluster_data = cluster_data.sample(n=sample_size, random_state=self.seed, replace=False)
            self._new_data = pd.concat([self._new_data, cluster_data], axis=0)

        self._new_data.reset_index(drop=True, inplace=True)
        self._new_data.drop(columns='cluster', inplace=True)

        # Density-Based Undersampling
    def density_based_undersample(self, k=20, reduction_factor=0.5):

        # Step 1: Compute mean distance to k nearest neighbors for each sample

        sorted_distances = np.sort(self._distance_matrix, axis=1)
        mean_distances = np.mean(sorted_distances[:, 1:k+1], axis=1)

        # Step 2: Compute density as inverse of mean distance
        density = 1 / (mean_distances + 1e-5)  # Add epsilon to avoid division by zero

        # Step 3: Add density to the DataFrame
        df = self._original_data.copy(deep=True)
        df['density'] = density

        # Step 4: Determine density threshold
        density_threshold_percentile = 70
        threshold = np.percentile(df['density'], density_threshold_percentile)

        # Step 5: Split into high-density and low-density samples
        high_density = df[df['density'] > threshold]
        low_density = df[df['density'] <= threshold]

        # Step 6: Calculate number of high-density samples to remove
        n_remove = int(len(high_density) * reduction_factor)

        # Step 7: Randomly remove samples from high-density group
        high_density_reduced = high_density.sample(n=len(high_density) - n_remove, random_state=42)

        # Step 8: Combine low-density and reduced high-density samples
        under_sampled_df = pd.concat([low_density, high_density_reduced], axis=0).reset_index(drop=True)

        self._new_data = under_sampled_df.drop(columns='density')




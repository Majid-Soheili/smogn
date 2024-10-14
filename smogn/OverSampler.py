
import pandas as pd
import numpy as np
import random
import logging
from tqdm import tqdm
from smogn.Schema import Schema
from smogn.box_plot_stats import box_plot_stats

"""
This module contains the implementation of the SMOGN-R algorithm.
We assume that all column should be numeric. If there are nominal columns, they should be transformed to integers. 
"""

class OverSampler:
    def __init__(self, data, index, cat_columns, percentage:float, perturbation, nk=5, seed = 0, verbose = False):
        """
        :param data (pd.DataFrame): The dataset to be used for over-sampling.
        :param index (pandas.index): The index which make a subset of data frame.
        :param cat_columns (list): The list of column names that are categorical.
        :param percentage (float): The percentage of under/over-sampling to be performed.
        :param perturbation (float): The amount of noise to add original samples to generate synthetic samples.
        :param nk (int, optional): The number of nearest neighbors for  oversampling. Default is 5.
        :param seed (int, optional): The random seed for reproducibility. Default is 0.
        :param verbose (bool, optional): Controls the verbosity of the output. Default is False.
        """

        np.random.seed(seed)
        random.seed(seed)

        self._original_data = data.iloc[index, :].copy(deep=True)
        self._original_data.reset_index(drop=True, inplace=True)
        self._categorical_columns = cat_columns
        self._percentage = percentage
        self._perturbation = perturbation
        self._nk = nk

        self._synth_data = pd.DataFrame()
        self._distance_matrix = None
        self._max_distance = None
        self._knn_matrix = None
        self._base_obs_indices = None # index of observation data used as the base to generate synthetic data
        self._schema = None

        self._seed = seed
        self._verbose = verbose

        # Initialize the schema
        self._remove_nan_rows()
        self._define_schema()

        # Initialize the pre-processing
        self._compute_distance_matrix()
        self._compute_nn_matrix()
        self._calculate_max_distance_threshold()
        self._define_base_obs_index()


    ## == Public methods == ##

    def generate_synthetic_data(self):
        # Generate synthetic data
        # For each base observation, generate nk synthetic observations
        # For each synthetic observation

        synth_list = []
        for obs in tqdm(self._base_obs_indices, ascii=True, desc="synth_matrix", disable=not self._verbose):

            neighbour_indic, safe = self._choose_random_nearest_neighbour(obs)

            if safe:
                synth = self._generate_synthetic_smoter(obs, neighbour_indic)
            else:
                synth = self._generate_synthetic_gaussian(obs)

            synth_list.append(synth)

        self._synth_data = pd.DataFrame(synth_list, columns=self._schema.column_names)
        self._reconstruct_synth_schema()
        return self._synth_data

    ## == Private methods == ##

    def _compute_distance_matrix(self):

        # Compute the distance matrix
        # As a rule, the columns term refer to all columns in the data frame including the target column
        # and the features term refer to all columns in the data frame except the target column


        data_num_array = self._original_data[self._schema.numerical_columns].to_numpy()
        data_nom_array = self._original_data[self._schema.nominal_columns].to_numpy()
        range_num = self._schema.column_range_values[self._schema.numerical_columns_mask]


        if self._schema.numerical_columns_count > 0 and np.any(range_num == 0):
            index = np.where(range_num == 0)[0]
            cname = self._schema.column_names[index]
            logging.warning(f"Warning: ranges_num contains zero values at indices {index} - {cname}.")
            range_num += 1e-8  # Add a small value to avoid division by zero


        # The number of constant columns is not important for the distance computation
        if self._schema.numerical_columns_count > 0 and self._schema.nominal_columns_count == 0:
            # Case 1: All features are numeric
            # Compute Euclidean distance using vectorized operations
            s = np.sum(data_num_array ** 2, axis=1)
            self._distance_matrix = np.sqrt(
                s[:, np.newaxis] + s[np.newaxis, :] - 2 * np.dot(data_num_array, data_num_array.T)
            )

        elif self._schema.nominal_columns_count > 0 and self._schema.numerical_columns_count == 0:
            # Case 2: All features are nominal
            # Compute Hamming distance using vectorized operations
            diff_nom = data_nom_array[:, np.newaxis, :] != data_nom_array[np.newaxis, :, :]
            self._distance_matrix = np.sum(diff_nom, axis=2).astype(float)

        elif self._schema.numerical_columns_count > 0 and self._schema.nominal_columns_count > 0:
            # Case 3: Mixed features (both numeric and nominal)
            # Numeric part
            diff_num = (data_num_array[:, np.newaxis, :] - data_num_array[np.newaxis, :, :]) / range_num  # Normalize differences

            diff_num **= 2  # Square differences
            sum_diff_num = np.sum(diff_num, axis=2)

            # Nominal part
            diff_nom = data_nom_array[:, np.newaxis, :] != data_nom_array[np.newaxis, :, :]
            # diff_nom = diff_nom.astype(float)
            diff_nom = np.where(diff_nom, 1.0, 0.0)
            sum_diff_nom = np.sum(diff_nom, axis=2)

            # Combine numeric and nominal distances
            self._distance_matrix = np.sqrt(sum_diff_num + sum_diff_nom)
        else:
            # No features present
            self._distance_matrix = None
            raise ValueError("No features present in the data.")

    def _compute_nn_matrix(self):
        # Compute the nearest neighbors matrix
        self._knn_matrix = np.argsort(self._distance_matrix, axis=1)[:, 1:self._nk + 1]

    def _calculate_max_distance_threshold(self):
        ## calculate max distances to determine if gaussian noise is applied
        ## (half the median of the distances per observation)

        n = self._original_data.shape[0]
        max_dist = [None] * n
        for i in range(n):

            max_dist[i] = box_plot_stats(self._distance_matrix[i])["stats"][2] / 2

        #self._max_distance = np.median(np.sort(self._distance_matrix)[:, -self._nk // 2])
        self._max_distance = max_dist

    def _define_base_obs_index(self):

       # Define the index of observations that will be used as bases to generate synthetic data.
       # If the percentage is less than 1, then randomly select base observations.
       # If the percentage is greater than 1, all observations will be used as bases.

       num_synth_per_obs = 0
       num_total_obs = self._original_data.shape[0]

       if self._percentage < 1:
           num_rand_obs = int(num_total_obs * self._percentage)
       else:
           num_synth_per_obs = int(self._percentage - 1)
           num_rand_obs = int(num_total_obs * (self._percentage - 1 - num_synth_per_obs))

       # The index of the original dataset was reset in the initializing class,
       # thus we can use a simple range as the index.

       indices = []
       for i in range(num_synth_per_obs):
           indices.append(np.array(range(num_total_obs)))

       indices.append(np.random.choice(range(num_total_obs), size = num_rand_obs, replace=False))
       self._base_obs_indices = np.concatenate(indices)

    def _generate_synthetic_smoter(self, obs, neigh):
        # Generate synthetic data using the SMOTR algorithm
        ## conduct synthetic minority over-sampling
        ## technique for regression (smoter)

        observation = self._original_data.iloc[obs, :]
        neighbour = self._original_data.iloc[neigh, :]
        diffs = neighbour - observation
        rate = np.random.random()
        synth = observation + rate * diffs

        ## randomly assign nominal / categorical features from
        ## observed cases and selected neighbor
        for col in self._schema.nominal_columns:
            synth[col] = observation[col] if np.random.random() < 0.5 else neighbour[col]

        ## generate synthetic y response variable by
        ## inverse distance weighted
        synth_target = self._inverse_distance_weighted(obs, neigh, synth)
        synth[self._schema.target_name] = synth_target

        return synth

    def _generate_synthetic_gaussian(self, obs):
        # Generate synthetic data using Gaussian noise
        ## conduct synthetic minority over-sampling technique
        ## for regression with the introduction of gaussian
        ## noise (smoter-gn)

        t_per = min(self._max_distance[obs], self._perturbation)
        synth = self._original_data.iloc[obs, :].copy(deep=True)

        # rate for each column is different and based on the std of the column
        # for numerical columns, the rate is a random number from a normal distribution
        # with mean 0 and std of corresponding column
        # for nominal columns, the rate is a random number from a normal distribution
        # with mean 0 and std of 1

        for i, col in enumerate(self._schema.column_names):
            rate = np.random.normal(loc = 0, scale = self._schema.column_std_values[i], size=1) * t_per
            synth[col] = synth[col] + rate

        for col in self._schema.nominal_columns:
            unique_values = self._schema.nominal_unique_values[col]
            unique_probs = self._schema.nominal_unique_probabilities[col]
            selected_value = np.random.choice(unique_values, p=unique_probs, size=1)[0]
            synth[col] = selected_value

        return synth

    def _choose_random_nearest_neighbour(self, obs):
        # for a given observation indic,
        # 1) choose a random neighbour from the nk nearest neighbours
        # 2) specify if the selected neighbour is safe or not.
        # safe means that the distance between the observation and the selected neighbour is
        # less than the max distance threshold

        # select random number from 1 to nk
        random_neighbour = random.randint(0, self._nk - 1)
        random_neighbour_indices = self._knn_matrix[obs, random_neighbour]
        safe = self._distance_matrix[obs, random_neighbour_indices] < self._max_distance[obs]
        return [random_neighbour_indices, safe]

    def _inverse_distance_weighted(self, obs, neigh, synth:pd.Series):
        # Generate synthetic y response variable by inverse distance weighted

        # Calculate the difference for all columns except the target column
        observation = self._original_data.iloc[obs, :]
        neighbour = self._original_data.iloc[neigh, :]

        feature_name = self._schema.features_name
        diff_observation = abs(synth[feature_name] - observation[feature_name])
        diff_neighbour = abs(synth[feature_name] - neighbour[feature_name])

        # for nominal columns, the difference is 0 or 1
        for col in self._schema.nominal_columns:
            diff_observation[col] = 1 if synth[col] != observation[col] else 0
            diff_neighbour[col] = 1 if synth[col] != neighbour[col] else 0

        # the range for nominal columns is 1
        feature_range =self._schema.column_range_values[self._schema.features_index]
        diff_observation = diff_observation.to_numpy()/feature_range
        diff_neighbour = diff_neighbour.to_numpy()/feature_range


        diff_neighbour = sum(diff_neighbour)
        diff_observation = sum(diff_observation)

        neighbour_target = neighbour[self._schema.target_name]
        observation_target = observation[self._schema.target_name]

        if diff_neighbour == diff_observation:
            synth_target = (observation_target + neighbour_target) / 2
        else:
            synth_target = (observation_target * diff_neighbour + neighbour_target * diff_observation) / (diff_neighbour + diff_observation)

        return synth_target

    def _reconstruct_synth_schema(self):
        # Reconstruct the data
        self._checking_non_negative_columns()
        self._checking_constant_columns()

    def _checking_non_negative_columns(self):
        # Check teh columns for negative values which they should not have
        for col in self._schema.non_negative_columns:
            self._synth_data[col] = self._synth_data[col].apply(lambda x: x if x >= 0 else 0)

    def _checking_constant_columns(self):
        # Check the columns for constant values
        for col in self._schema.constant_features:
            self._synth_data[col] = self._schema.constant_values[col]

    def _remove_nan_rows(self):
        self._original_data.dropna(inplace=True)

    def _define_schema(self):
        self._schema = Schema()
        self._schema.define_schema(self._original_data, self._categorical_columns)


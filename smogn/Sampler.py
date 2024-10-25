import pandas as pd
import numpy as np
import random
import logging

from tqdm import tqdm
from smogn.Schema import Schema

class Sampler:
    def __init__(self, data, index, percentage:float, nk=5, seed = 0, verbose = False):

        """
        :param data (pd.DataFrame): The dataset to be used for over-sampling.
        :param index (pandas.index): The index which make a subset of data frame.
        :param percentage (float): The percentage of under/over-sampling to be performed.
        :param nk (int, optional): The number of nearest neighbors for  oversampling. Default is 5.
        :param seed (int, optional): The random seed for reproducibility. Default is 0.
        :param verbose (bool, optional): Controls the verbosity of the output. Default is False.
        """

        np.random.seed(seed)
        random.seed(seed)

        self._original_data = data.iloc[index, :].copy(deep=True)
        self._original_data.reset_index(drop=True, inplace=True)
        self.percentage = percentage
        self.num_new_data = int(self.percentage * len(self._original_data))
        self.seed = seed
        self.nk = nk
        self.verbose = verbose
        self._logger = logging.getLogger("Sampler")
        if not self._logger.hasHandlers():
            self._logger.addHandler(logging.StreamHandler())
            self._logger.setLevel(logging.DEBUG)

        self._new_data = pd.DataFrame()
        self._distance_matrix = None
        self._schema = None

        self._seed = seed
        self._verbose = verbose

        # Initialize the schema
        self._remove_nan_rows()
        self._define_schema()

        # Initialize the pre-processing
        self._categorical_to_int()
        self._compute_distance_matrix()



    # Private methods ===========================================

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
            diff_num = (data_num_array[:, np.newaxis, :] - data_num_array[np.newaxis, :, :]) / range_num  # Normalize differences
            diff_num **= 2  # Square differences
            sum_diff_num = np.sum(diff_num, axis=2)
            self._distance_matrix = np.sqrt(sum_diff_num)

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

    def _normalize_distance_matrix(self):
        # Create a mask to exclude diagonal elements

        mask = ~np.eye(self._distance_matrix.shape[0], dtype=bool)
        non_diagonal = self._distance_matrix[mask]

        dist_min = non_diagonal.min()  # Typically > 0
        dist_max = non_diagonal.max()
        norm_dist = (self._distance_matrix - dist_min) / (dist_max - dist_min)
        np.fill_diagonal(norm_dist, 0)
        self._distance_matrix = norm_dist

    def _remove_nan_rows(self):
        self._original_data.dropna(inplace=True)

    def _define_schema(self):
        self._schema = Schema()
        self._schema.define_schema(self._original_data)

    def _categorical_to_int(self):
        # Convert categorical columns to integers
        for col, unique_values in self._schema.nominal_unique_values.items():
            mapping = {k: v for v, k in enumerate(unique_values)}
            self._original_data[col] = self._original_data[col].map(mapping)
            self._original_data[col] = self._original_data[col].astype(int)

    def _int_to_categorical(self):
        # Convert integers to categorical columns
        for col, unique_values in self._schema.nominal_unique_values.items():
            mapping = {v: k for v, k in enumerate(unique_values)}
            self._new_data[col] = self._new_data[col].map(mapping)

    def _reconstruct_synth_schema(self):
        # Reconstruct the data
        self._int_to_categorical()
        self._checking_non_negative_columns()
        self._checking_constant_columns()

    def _checking_non_negative_columns(self):
        # Check teh columns for negative values which they should not have
        for col in self._schema.non_negative_columns:
            self._new_data[col] = self._new_data[col].apply(lambda x: x if x >= 0 else 0)

    def _checking_constant_columns(self):
        # Check the columns for constant values
        for col in self._schema.constant_features:
            self._new_data[col] = self._schema.constant_values[col]

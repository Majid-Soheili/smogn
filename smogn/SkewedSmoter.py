import sys
import numpy as np
import pandas as pd
import logging
from smogn.OverSampler import OverSampler
from smogn.UnderSampler import UnderSampler

class SkewedSmoter:
    def __init__(self, data, target, verbose=1, seed = 123):

        # General parameters
        self.data = data
        self.target = target

        self._logger = logging.getLogger("SkewedSmoter")
        self._logger.setLevel(logging.DEBUG if verbose > 0 else logging.WARNING)
        self._logger.propagate = False  # Prevent propagation to root logger

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            self._logger.addHandler(handler)

        self.synthetic_data = pd.DataFrame()
        self.steepness = 1
        self.skewness = 0.3
        self.focus = None
        self.bins = None
        self.synth_bins_populations = None
        self.bin_width = 0.5

        self.verbose = verbose
        self.seed = seed
        self.under_method = "cluster"
        np.random.seed(seed)


        self.feat_dtypes_orig = [self.data.iloc[:, j].dtype for j in range(self.data.shape[1])]
        self._remove_duplicated_rows()
        self._put_target_column_last()
        self._calculate_bins()

    def _init(self):

        # Data quality checks
        if self.data[self.target] is None or len(self.data[self.target]) == 0:
            raise ValueError("Target column is empty")

        self._calculate_bins()
        self._compute_bin_population()

    def generate_synthetic_data(self):

        self._compute_bin_population()
        synthetic_data_list = []

        for i in range(len(self.bins) - 1):

            index, rate = self._handle_bin_population(i)

            if index is None: # the bin is not valid for sampling
                continue

            if rate > 1: # Oversample the bin

                synthetic_data_list.append(self.data.iloc[index, :])
                over_sampler = OverSampler(self.data, index, percentage=rate, perturbation=0.02, nk=5, verbose=True, seed=self.seed)
                synth = over_sampler.generate_synthetic_data()

                if self.verbose > 0:
                    if synth is None or len(synth) == 0:
                        self._logger.debug(f"Generated 0 synthetic samples for bin {i}")
                    else:
                        self._logger.debug(f"Generated {synth.shape[0]} synthetic samples for bin {i}")

            else: # Undersample the bin

                under_sampler = UnderSampler(self.data, index, method= self.under_method, percentage=rate, seed=self.seed, verbose=self.verbose)
                synth = under_sampler.provide_under_sampled_data()

                if self.verbose > 0:
                    if synth is None or len(synth) == 0:
                        self._logger.debug(f"No samples selected for bin {i}")
                    else:
                        self._logger.debug(f"Selected {synth.shape[0]} samples for bin {i}")

            if synth is not None and len(synth) > 0:
                synthetic_data_list.append(synth)


        self.synthetic_data = pd.concat(synthetic_data_list, ignore_index=True)
        self._restore_original_data_types()
        return self.synthetic_data

    # Function to generate a skewed distribution using log-normal
    def skewed_distribution(self):
        if self.focus is None:
            raise ValueError("Focus index is not set")
        elif self.focus not in self.bins:
            raise ValueError("Focus index is not in the bins")

        findex = np.where(self.bins == self.focus)[0]
        x = np.arange(len(self.bins)) - findex
        # Use a log-normal distribution centered at the focus index
        distribution = np.exp(-self.steepness * np.abs(x) ** self.skewness)
        distribution /= distribution.sum()  # Normalize to sum to 1
        return distribution

    def _compute_bin_population(self):
        # Generate the population distribution based on the skewed function
        weights = self.skewed_distribution()
        population = self.data.shape[0]
        population_distribution = (weights * population).astype(int)
        self.synth_bins_populations = population_distribution

    # Private Methods ===============================================

    def _handle_bin_population(self, idx):
        lower_bound = self.bins[idx]
        upper_bound = self.bins[idx + 1]
        index = self.data[(self.data[self.target] >= lower_bound) & (self.data[self.target] < upper_bound)].index
        original_bin_population = len(index)
        synthetic_bin_population = self.synth_bins_populations[idx]

        if original_bin_population <= 5:
            self._logger.warning(f"Bin {idx} has less than 5 samples. Skipping...")
            return None, None
        else:
            rate = synthetic_bin_population / original_bin_population
            return index, rate

    def _restore_original_data_types(self):
        result_df = pd.DataFrame()
        d = len(self.data.columns)
        for j in range(d):
            # for category based on the original data type it can handle it
            dtype_orig = self.feat_dtypes_orig[j]
            column_name = self.synthetic_data.columns[j]
            column = self.synthetic_data[column_name]

            if dtype_orig in [np.int64, pd.Int64Dtype()]:
                column = column.round()

            column = column.astype(dtype_orig)
            result_df = pd.concat([result_df, column], axis=1)

        self.synthetic_data = result_df

    def _calculate_bins(self):

        min_value = self.data[self.target].min()
        min_value = int(round(min_value, 0))
        max_value = self.data[self.target].max()
        max_value = int(round(max_value, 0))
        step = self.bin_width
        self.bins = np.arange(min_value, max_value + step, step)

    def _remove_duplicated_rows(self):
        n_duplicates = self.data.duplicated().sum()
        if n_duplicates > 0:
            self._logger.warning(f"Removing {n_duplicates} duplicated rows")
        self.data = self.data.drop_duplicates()
        return self

    # a function to put target column in the last column
    def _put_target_column_last(self):
        cols = self.data.columns.tolist()
        cols.remove(self.target)
        cols.append(self.target)
        self.data = self.data[cols]
        return self

   # Public Getters and Setters ========================================
    def get_original_data(self):
        return self.data
    def get_synthetic_data(self):
        return self.synthetic_data

    def set_data(self, data):
        self.data = data
        return self

    def get_target(self):
        return self.target

    def set_target(self, target):
        self.target = target
        return self

    def get_steepness(self):
        return self.steepness

    def set_steepness(self, steepness):
        self.steepness = steepness
        return self

    def get_skewness(self):
        return self.skewness

    def set_skewness(self, skewness):
        self.skewness = skewness
        return self

    def get_focus(self):
        return self.focus

    def set_focus(self, focus):
        self.focus = focus
        return self

    def set_bin_width(self, bin_width):
        self.bin_width = bin_width
        return self

    def set_under_method(self, method):
        self.under_method = method
        return self
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from smogn.OverSampler import OverSampler


class TestOverSampler(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'feature2': [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            'category': [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            'constant': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'target': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })

        self.index = np.arange(len(self.data))
        self.cat_columns = ['category']
        self.percentage = 2.2
        self.perturbation = 0.1

        # Initialize OverSampler with mocked Schema
        self.oversampler = OverSampler(
            data=self.data,
            index=self.index,
            cat_columns=self.cat_columns,
            percentage=self.percentage,
            perturbation=self.perturbation,
            nk=2,  # Use smaller nk for testing
            seed=42,
            verbose=False
        )

    def test_initialization(self):
        """Test if OverSampler initializes correctly."""
        # Check if original_data is correctly set
        pd.testing.assert_frame_equal(self.oversampler._original_data, self.data.reset_index(drop=True))

        # Check categorical columns
        self.assertEqual(self.oversampler._categorical_columns, self.cat_columns)

        # Check percentage and perturbation
        self.assertEqual(self.oversampler._percentage, self.percentage)
        self.assertEqual(self.oversampler._perturbation, self.perturbation)

        # Check nk
        self.assertEqual(self.oversampler._nk, 2)

        # Check seed
        self.assertEqual(self.oversampler._seed, 42)

        # Check verbose
        self.assertFalse(self.oversampler._verbose)

    def test_distance_matrix_computation(self):
        """Test the distance matrix computation."""
        # Access the private method _compute_distance_matrix
        with patch.object(self.oversampler, '_remove_nan_rows'), \
                patch.object(self.oversampler, '_define_schema'):
            
            expected_distance = np.array([
                [0.000, 1.018, 0.385, 1.155, 0.770, 1.387, 1.155, 1.680, 1.836, 2.000],
                [1.018, 0.000, 1.024, 0.385, 1.155, 0.770, 1.387, 1.155, 1.347, 1.541],
                [0.385, 1.024, 0.000, 1.024, 0.385, 1.155, 0.770, 1.387, 1.528, 1.678],
                [1.155, 0.385, 1.024, 0.000, 1.024, 0.385, 1.155, 0.770, 0.962, 1.155],
                [0.770, 1.155, 0.385, 1.024, 0.000, 1.024, 0.385, 1.155, 1.261, 1.387],
                [1.387, 0.770, 1.155, 0.385, 1.024, 0.000, 1.024, 0.385, 0.577, 0.770],
                [1.155, 1.387, 0.770, 1.155, 0.385, 1.024, 0.000, 1.024, 1.072, 1.155],
                [1.680, 1.155, 1.387, 0.770, 1.155, 0.385, 1.024, 0.000, 0.192, 0.385],
                [1.836, 1.347, 1.528, 0.962, 1.261, 0.577, 1.072, 0.192, 0.000, 0.192],
                [2.000, 1.541, 1.678, 1.155, 1.387, 0.770, 1.155, 0.385, 0.192, 0.000]
            ])

            expected_distance = np.round(expected_distance, 2)
            np.testing.assert_array_almost_equal(self.oversampler._distance_matrix, expected_distance, decimal=2)

    def test_compute_nn_matrix(self):
        """Test the computation of the nearest neighbors matrix."""
        expected_knn_matrix = np.array([
            [2, 4],
            [3, 5],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 2],
            [8, 5],
            [7, 9],
            [8, 7]
        ])
        # the distance matrix will be default of the dataset define in the setUp
        ## self.oversampler._distance_matrix = controlled_distance_matrix
        self.oversampler._nk = 2  # Ensure nk is 2
        self.oversampler._compute_nn_matrix()

        # Assert that _knn_matrix is as expected
        np.testing.assert_array_equal(self.oversampler._knn_matrix, expected_knn_matrix)

    def test_calculate_max_distance_threshold(self):
        """Test the calculation of the maximum distance threshold."""
        expected_values = np.array([0.5775, 0.54475, 0.512, 0.4965, 0.512, 0.385, 0.524, 0.4485, 0.5085, 0.5775])
        np.testing.assert_array_almost_equal(self.oversampler._max_distance, expected_values, decimal=2)

    def test_reconstruct_synth_schema_negative_values(self):
        """Test that non-negative columns are correctly handled in synthetic data."""
        with patch.object(self.oversampler, '_choose_random_nearest_neighbour') as mock_choose_nn, \
                patch.object(self.oversampler, '_generate_synthetic_smoter') as mock_generate_smoter:
            # Setup the mock for _choose_random_nearest_neighbour
            mock_choose_nn.return_value = [1, True]  # Always choose neighbor at index 1 and safe

            # Setup the mock for _generate_synthetic_smoter with potential negative values
            synthetic_sample = pd.Series({
                'feature1': -1.5,  # Negative value, should be set to 0
                'feature2': 9.5,
                'category': 1.0,
                'constant': 2.0,   # Constant column should be keep as 1
                'target': 15.0
            })
            mock_generate_smoter.return_value = synthetic_sample

            # Call the method under test
            synth_data = self.oversampler.generate_synthetic_data()

            # Assertions
            self.assertTrue((synth_data['feature1'] >= 0).all())
            # 'constant' column should remain unchanged
            self.assertTrue((synth_data['constant'] == 1).all())


    def test_generate_synthetic_smoter(self):
        """Test the _generate_synthetic_smoter method for correct synthetic sample generation."""
        # Define base observation index and neighbor index
        obs_index = 0  # First observation
        neigh_index = 1  # Second observation

        # Expected behavior:
        # observation = data.iloc[0]
        # neighbor = data.iloc[1]
        # diffs = neighbor - observation = [1.0, -1.0, 1, 0, 10]
        # rate = mocked np.random.random() = 0.5
        # synth = observation + rate * diffs = [1.0 + 0.5*1.0, 10.0 + 0.5*(-1.0), 0 or 1, 1, ...]
        # For categorical 'category', randomly choose between 0 and 1
        # For target, compute inverse distance weighted (mocked or computed based on diffs)

        # Mock np.random.random() to return a fixed rate
        with patch('numpy.random.random', return_value=0.5), \
                patch('random.random', return_value=0.3), \
                patch('random.choice', return_value=1):  # Mock category assignment

            # Call the private method _generate_synthetic_smoter
            synthetic_sample = self.oversampler._generate_synthetic_smoter(obs=obs_index, neigh=neigh_index)
            synthetic_sample = synthetic_sample.to_numpy()

            expected_synth = pd.Series({
                'feature1': 1.5,
                'feature2': 9.5,
                'category': 1,
                'constant': 1,
                'target': 19.09
            })

            expected_synth = expected_synth.to_numpy()
            # Assertions
            np.testing.assert_array_almost_equal(synthetic_sample, expected_synth, decimal=2)

    def test_generate_synthetic_gaussian(self):
        """Test the _generate_synthetic_gaussian method for correct synthetic sample generation."""
        # Define base observation index and neighbor index
        # This test function should be considered more.
        obs_index = 0   # First observation

        with patch('numpy.random.random', return_value=0.5), \
                patch('random.choice', return_value=1):  # Mock category assignment
            # Call the private method _generate_synthetic_gaussian
            synthetic_sample = self.oversampler._generate_synthetic_gaussian(obs=obs_index)

            expected_synth = pd.Series({
                'feature1': 1.08,
                'feature2': 10.31,
                'category': 0,
                'constant': 1,
                'target': 8.27
            })

            expected_synth = expected_synth.to_numpy()
            # Assertions
            np.testing.assert_array_almost_equal(synthetic_sample, expected_synth, decimal=2)

import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import seaborn as sns
import matplotlib.pyplot as plt
from smogn.SkewedSmoter import SkewedSmoter


class TestSkewedSmoter(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset with multiple features and a continuous target
        np.random.seed(42)  # For reproducibility
        self.sample_data = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 100),
            'feature2': np.random.uniform(0, 1, 100),
            'target': np.random.lognormal(mean=1.0, sigma=0.5, size=100)  # Adjust mean and sigma as needed
        })
        self.target_column = 'target'
        self.smoter = SkewedSmoter(data=self.sample_data, target=self.target_column)

    def test_initialization(self):
        # Test if the object initializes correctly
        self.assertTrue(self.smoter.get_data().equals(self.sample_data))
        self.assertEqual(self.smoter.get_target(), self.target_column)
        self.assertEqual(self.smoter.get_steepness(), 2)
        self.assertEqual(self.smoter.get_skewness(), 0.5)
        self.assertEqual(self.smoter.get_focus(), 0)
        # After _calculate_bins, bins should be a numpy array
        expected_bins = 17
        self.assertEqual(self.smoter.bins.size, expected_bins)

    def test_skewed_distribution_sum(self):
        # Test if the skewed distribution sums to 1
        distribution = self.smoter.skewed_distribution()
        self.assertAlmostEqual(distribution.sum(), 1.0, places=5)

    def test_skewed_distribution_length(self):
        # Test if the distribution length matches the number of bins
        distribution = self.smoter.skewed_distribution()
        self.assertEqual(len(distribution), len(self.smoter.bins))

    def test_skewed_distribution_values(self):
        # Test if all distribution values are positive
        distribution = self.smoter.skewed_distribution()
        self.assertTrue(np.all(distribution >= 0))

        # Optionally, check if the distribution is skewed as expected
        # For example, highest weight should be around the focus
        max_index = np.argmax(distribution)
        expected_focus_bin = self.smoter.focus
        # Since bins are a range, the focus corresponds to index self.focus
        # Adjust if focus is not within the bin range
        if 0 <= expected_focus_bin < len(distribution):
            self.assertEqual(max_index, self.smoter.focus)
        else:
            # If focus is out of range, check if max_index is at one end
            self.assertIn(max_index, [0, len(distribution) - 1])

    def test_compute_bin_population_sum(self):
        # Test if the bin population sums to the total population
        population_distribution = self.smoter._compute_bin_population()
        max_diff = len(self.smoter.bins)
        original_population = self.smoter.get_data().shape[0]
        synth_population = population_distribution.sum()
        self.assertTrue(abs(original_population - synth_population) <= max_diff)

    def test_compute_bin_population_type(self):
        # Test if the population distribution contains integers
        population_distribution = self.smoter._compute_bin_population()
        self.assertTrue(np.issubdtype(population_distribution.dtype, np.integer))

    def test_getters_setters(self):
        # Test getters and setters for data
        new_data = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 50),
            'feature2': np.random.uniform(0, 1, 50),
            'target': np.random.uniform(0, 8, 50)
        })
        self.smoter.set_data(new_data)
        self.assertTrue(self.smoter.get_data().equals(new_data))

        # Test getters and setters for target
        new_target = 'new_target'
        new_data[new_target] = np.random.uniform(0, 8, 50)
        self.smoter.set_target(new_target)
        self.assertEqual(self.smoter.get_target(), new_target)

        # Test getters and setters for steepness
        new_steepness = 3
        self.smoter.set_steepness(new_steepness)
        self.assertEqual(self.smoter.get_steepness(), new_steepness)

        # Test getters and setters for skewness
        new_skewness = 1.0
        self.smoter.set_skewness(new_skewness)
        self.assertEqual(self.smoter.get_skewness(), new_skewness)

        # Test getters and setters for focus
        new_focus = 2
        self.smoter.set_focus(new_focus)
        self.assertEqual(self.smoter.get_focus(), new_focus)

    def test_calculate_bins(self):
        # Test the _calculate_bins method with specific data
        # Modify the target to have a known range
        self.smoter.data[self.target_column] = np.repeat([1.0, 2.5, 3.0, 4.5, 5.0], 20)
        self.smoter._calculate_bins()
        expected_bins = np.arange(1, 5 + 0.5, 0.5)
        np.testing.assert_array_equal(self.smoter.bins, expected_bins)

    def test_skewed_distribution_after_setting_parameters(self):
        # Test if skewed_distribution changes after setting new parameters
        self.smoter.set_steepness(3)
        self.smoter.set_skewness(1.0)
        self.smoter.set_focus(2)
        distribution = self.smoter.skewed_distribution()
        self.assertAlmostEqual(distribution.sum(), 1.0, places=5)
        self.assertEqual(len(distribution), len(self.smoter.bins))
        # Check if the peak shifted to the new focus
        if 0 <= 2 < len(distribution):
            max_index = np.argmax(distribution)
            self.assertEqual(max_index, 2)
        else:
            # If focus index 2 is out of range, ensure max is at expected position
            self.assertIn(np.argmax(distribution), [0, len(distribution) - 1])

    def test_skewed_distribution_with_single_bin(self):
        # Test behavior when there is only one unique bin
        single_bin_data = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 10),
            'feature2': np.random.uniform(0, 1, 10),
            'target': [4.0] * 10  # Only one unique bin
        })
        smoter_single = SkewedSmoter(data=single_bin_data, target='target')
        smoter_single._calculate_bins()
        distribution = smoter_single.skewed_distribution()
        self.assertEqual(len(distribution), 1)
        self.assertEqual(distribution[0], 1.0)

    def test_skewed_distribution_with_no_data(self):
        # Test behavior when data is empty
        empty_data = pd.DataFrame(columns=['feature1', 'feature2', 'target'])
        with self.assertRaises(ValueError):
            SkewedSmoter(data=empty_data, target='target')

    def test_set_data_updates_bins(self):
        # Test if setting new data updates the number of bins correctly
        new_data = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 50),
            'feature2': np.random.uniform(0, 1, 50),
            'target': np.random.uniform(0, 8, 50)  # Continuous target in [0,8]
        })
        self.smoter.set_data(new_data)
        self.smoter._calculate_bins()
        expected_bins = 17
        self.assertEqual(len(self.smoter.bins), expected_bins)

    def test_set_target_updates_bins(self):
        # Test if setting a new target updates the number of bins correctly
        new_data = pd.DataFrame({
            'feature1': np.random.uniform(0, 1, 50),
            'feature2': np.random.uniform(0, 1, 50),
            'new_target': np.random.uniform(0, 8, 50)  # Continuous target in [0,8]
        })
        self.smoter.set_data(new_data).set_target('new_target')
        expected_bins = 17
        self.assertEqual(len(self.smoter.bins), expected_bins)

    def test_data_balancing(self):
        before_population = self.smoter.get_data()[self.target_column].copy(deep=True)
        new_data = self.smoter.generate_synthetic_data()
        after_population = new_data[self.target_column].copy(deep=True)

        # Plot ===============
        sns.kdeplot(before_population, label="Original")
        sns.kdeplot(after_population, label="Modified")
        plt.title('KDE Plot of Target Variable')
        plt.xlabel('Target')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
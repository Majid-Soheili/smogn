import unittest
import numpy as np
import pandas as pd
from smogn.Schema import Schema

class TestSchema(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data_numeric = pd.DataFrame({
            'A': [0.5, 1.5, 2.5, 3.5],
            'B': [1, 2, 1, 2],
            'C': [10, 10, 10, 10],       # constant column
            'D': [5, -1, 3, 2],          # contains negative value
            'T': [1.7, 2.3, 3.1, 3.5]    # Target column
        })

        self.data_mixed = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],   # non-numeric column
            'T': [1.7, 2.3, 3.1]    # Target column
        })

        self.data_constant = pd.DataFrame({
            'A': [1, 1, 1],
            'B': [2.0, 2.0, 2.0],
            'T': [1.7, 2.3, 3.1]
        })

        self.nominal_columns = ['B']

    def test_define_schema_numeric(self):
        schema = Schema()
        schema.define_schema(self.data_numeric, self.nominal_columns)

        # Test names
        np.testing.assert_array_equal(schema.features_name, np.array(['A', 'B', 'C', 'D']))
        np.testing.assert_array_equal(schema.column_names, np.array(['A', 'B', 'C', 'D', 'T']))

        # Test indexes
        np.testing.assert_array_equal(schema.features_index, np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(schema.column_indexes, np.array([0, 1, 2, 3, 4]))

        # Test target
        np.testing.assert_array_equal(schema.target_name, 'T')
        np.testing.assert_array_equal(schema.target_index, 4)


        # Test data_types
        expected_dtypes = np.array(['float64', 'int64', 'int64', 'int64', 'float64'])
        np.testing.assert_array_equal(schema.data_types, expected_dtypes)

        # Test nominal_columns
        np.testing.assert_array_equal(schema.nominal_features, np.array(['B']))

        # Test constant_columns
        np.testing.assert_array_equal(schema.constant_features, np.array(['C']))

        # Test numerical_columns
        np.testing.assert_array_equal(schema.numerical_features, np.array(['A', 'D']))

        # Test range_values
        expected_range_values = np.array([
            (3.0),      # A
            (1),        # B (nominal)
            (0),        # C
            (6)         # D
        ], dtype='float64')

        self.assertEqual(schema.feature_range_values.dtype, expected_range_values.dtype)
        self.assertEqual(schema.feature_range_values.shape, expected_range_values.shape)
        self.assertTrue(np.array_equal(schema.feature_range_values, expected_range_values))

        # Test non_negative_columns
        np.testing.assert_array_equal(schema.non_negative_columns, np.array(['A', 'B', 'C', 'T']))

    def test_define_schema_non_numeric(self):
        schema = Schema()
        with self.assertRaises(ValueError) as context:
            schema.define_schema(self.data_mixed, self.nominal_columns)
        self.assertIn("All columns in the data should be numeric.", str(context.exception))

    def test_constant_columns(self):
        # All columns are constant
        schema = Schema()
        schema.define_schema(self.data_constant, nominal_features=[])

        np.testing.assert_array_equal(schema.constant_features, np.array(['A', 'B']))
        np.testing.assert_array_equal(schema.numerical_features, np.array([]))
        expected_range_values = np.array([
            (0.0),
            (0.0)
        ], dtype='float64')

        self.assertEqual(schema.feature_range_values.dtype, expected_range_values.dtype)
        self.assertEqual(schema.feature_range_values.shape, expected_range_values.shape)
        self.assertTrue(np.allclose(schema.feature_range_values, expected_range_values))

    def test_non_negative_columns(self):
        data = pd.DataFrame({
            'A': [0, 1, 2],
            'B': [-1, 0, 1],
            'C': [5, 5, 5],
            'T': [1.7, 2.3, 3.1]
        })
        schema = Schema()
        schema.define_schema(data, nominal_features=[])

        np.testing.assert_array_equal(schema.non_negative_columns, np.array(['A', 'C', 'T']))

    def test_nominal_columns_set_correctly(self):
        schema = Schema()
        schema.define_schema(self.data_numeric, nominal_features=self.nominal_columns)

        np.testing.assert_array_equal(schema.nominal_features, np.array(['B']))
        np.testing.assert_array_equal(schema.numerical_features, np.array(['A', 'D']))

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, verbosity=2)
import unittest
import numpy as np
import pandas as pd
from smogn.Schema import Schema

class TestSchema(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data_numeric = pd.DataFrame({
            'A': [0.5, 1.5, 2.5, 3.5],
            'B': [1, 2, 1, 1],           # nominal column
            'C': [10, 10, 10, 10],       # constant column
            'D': [5, -1, 3, 2],          # contains negative value
            'T': [1.7, 2.3, 3.1, 3.5]    # Target column
        })
        self.data_numeric['B'] = self.data_numeric['B'].astype('category')

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

        self.data_nominal = pd.DataFrame({
            'A': [0.5, 1.5, 2.5, 3.5, 4.5], # numerical column
            'B': [2, 1, 1, 1, 2],           # nominal column and category data type
            'C': ['B', 'B', 'A', 'B', 'A'], # nominal column and category data type
            'D': [1, 0, 1, 0, 0],           # nominal column and object data type
            'T': [1.7, 2.3, 3.1, 4.5, 5.6]
        })
        self.data_nominal['B'] = self.data_nominal['B'].astype('category')
        self.data_nominal['C'] = self.data_nominal['C'].astype('category')
        self.data_nominal['D'] = self.data_nominal['D'].astype('object')


    def test_define_schema_numeric(self):
        schema = Schema()
        schema.define_schema(self.data_numeric)

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
        expected_dtypes = np.array(['float64', 'category', 'int64', 'int64', 'float64'])
        np.testing.assert_array_equal(schema.data_types, expected_dtypes)

        # Test nominal_columns
        np.testing.assert_array_equal(schema.nominal_features, np.array(['B']))

        # Test constant_columns
        np.testing.assert_array_equal(schema.constant_features, np.array(['C']))

        # Test numerical_columns
        np.testing.assert_array_equal(schema.numerical_features, np.array(['A', 'D']))

        # Test range_values
        # For avoiding division by zero, we set the constant and nominal range value to 1.0
        expected_range_values = np.array([
            (3.0),      # A
            (1.0),      # B (nominal)
            (1.0),      # C
            (6)         # D
        ], dtype='float64')

        self.assertEqual(schema.feature_range_values.dtype, expected_range_values.dtype)
        self.assertEqual(schema.feature_range_values.shape, expected_range_values.shape)
        self.assertTrue(np.array_equal(schema.feature_range_values, expected_range_values))

        # Test non_negative_columns
        np.testing.assert_array_equal(schema.non_negative_columns, np.array(['A', 'T']))

#    def test_define_schema_non_numeric(self):
#        schema = Schema()
#        with self.assertRaises(ValueError) as context:
#            schema.define_schema(self.data_mixed, None)
#        self.assertIn("All columns in the data should be numeric.", str(context.exception))

    def test_constant_columns(self):
        # All columns are constant
        schema = Schema()
        schema.define_schema(self.data_constant)

        np.testing.assert_array_equal(schema.constant_features, np.array(['A', 'B']))
        np.testing.assert_array_equal(schema.numerical_features, np.array([]))
        # For avoiding division by zero, we set the constant range value to 1.0
        expected_range_values = np.array([
            (1.0),
            (1.0)
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
        schema.define_schema(data)

        np.testing.assert_array_equal(schema.non_negative_columns, np.array(['A', 'T']))

    def test_nominal_columns_set_correctly(self):
        schema = Schema()
        schema.define_schema(self.data_nominal)

        np.testing.assert_array_equal(schema.nominal_features, np.array(['B', 'C', 'D']))
        np.testing.assert_array_equal(schema.numerical_columns, np.array(['A', 'T']))
        np.testing.assert_array_equal(schema.numerical_features, np.array(['A']))

        # test nominal_unique_values
        expected_nominal_unique_values = {
            'B': np.array([1, 2]),
            'C': np.array(['A', 'B']),
            'D': np.array([0, 1])
        }
        self.assertEqual(np.array_equal(schema.nominal_unique_values['B'], expected_nominal_unique_values['B']), True)
        self.assertEqual(np.array_equal(schema.nominal_unique_values['C'], expected_nominal_unique_values['C']), True)
        self.assertEqual(np.array_equal(schema.nominal_unique_values['D'], expected_nominal_unique_values['D']), True)


        # test nominal_unique_probabilities
        expected_nominal_unique_probabilities = {
            'B': np.array([0.6, 0.4]),
            'C': np.array([0.4, 0.6]),
            'D': np.array([0.6, 0.4])
        }
        self.assertEqual(np.array_equal(schema.nominal_unique_probabilities['B'], expected_nominal_unique_probabilities['B']), True)
        self.assertEqual(np.array_equal(schema.nominal_unique_probabilities['C'], expected_nominal_unique_probabilities['C']), True)
        self.assertEqual(np.array_equal(schema.nominal_unique_probabilities['D'], expected_nominal_unique_probabilities['D']), True)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, verbosity=2)
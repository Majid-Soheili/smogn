from dataclasses import dataclass, field
import numpy as np
@dataclass
class Schema:
    # Basic Members ============================================================

    __num_dtypes = ["int64", "float64"]
    target_name:str = ""
    target_index:int = 0
    features_name:np.ndarray = field(default_factory=lambda: np.array([]))
    features_index:np.ndarray = field(default_factory=lambda: np.array([]))
    column_indexes:np.ndarray = field(default_factory=lambda: np.array([]))
    column_names:np.ndarray = field(default_factory=lambda: np.array([]))
    data_types:np.ndarray = field(default_factory=lambda: np.array([]))

    # Range and Standard Deviation Members =====================================

    range_values:np.ndarray = field(default_factory=lambda: np.array([]))
    std_values:np.ndarray = field(default_factory=lambda: np.array([]))

    # Nominal Members ==========================================================

    nominal_columns:np.ndarray = field(default_factory=lambda: np.array([]))
    nominal_mask:np.ndarray = field(default_factory=lambda: np.array([]))
    nominal_columns_count:int = 0

    # nominal_unique_values should be a dictionary such that the key is the column name
    # and the value is the unique values for the column
    nominal_unique_values:dict = field(default_factory=lambda: {})

    # nominal unique probabilities should be a dictionary such that the key is the column name
    # and the value is the unique probabilities for the column
    nominal_unique_probabilities:dict = field(default_factory=lambda: {})

    # Numerical Members ========================================================

    numerical_columns:np.ndarray = field(default_factory=lambda: np.array([]))
    numerical_mask:np.ndarray = field(default_factory=lambda: np.array([]))
    numerical_columns_count:int = 0

    # Constant Members =========================================================

    constant_columns:np.ndarray = field(default_factory=lambda: np.array([]))
    constant_mask:np.ndarray = field(default_factory=lambda: np.array([]))

    # constant values should be a dictionary such that the key is the column name
    # and the value is the constant value for the column
    constant_values:dict = field(default_factory=lambda: {})
    constant_columns_count:int = 0

    # Non-negative Members =====================================================

    non_negative_columns:np.ndarray = field(default_factory=lambda: np.array([]))
    non_negative_mask:np.ndarray = field(default_factory=lambda: np.array([]))
    non_negative_columns_count:int = 0

    def define_schema(self, data, nominal_columns):

        self.names = np.array(data.columns)
        self.indexes = np.arange(len(self.names))
        self.data_types = np.array(data.dtypes)

        # We assume all columns are numeric
        if not all([dtype in self.__num_dtypes for dtype in self.data_types]):
            raise ValueError("All columns in the data should be numeric.")

        # 02 - Categorical columns handling ========================================
        # We assume that categorical are specified by the user
        self.nominal_columns = np.array(nominal_columns)

        # Create a boolean mask for nominal columns
        self.nominal_mask = np.isin(self.names, self.nominal_columns)

        self.nominal_columns_count = len(self.nominal_columns)

        self.nominal_unique_values = {col: data[col].unique() for col in self.nominal_columns}

        self.nominal_unique_probabilities = {col: data[col].value_counts(normalize=True) for col in self.nominal_columns}

        # 03 - Constant columns handling ===========================================
        # constant columns, each column that has only one unique value
        self.constant_columns = np.array([col for col in self.names if len(data[col].unique()) == 1])

        # Create a boolean mask for constant columns
        self.constant_mask = np.isin(self.names, self.constant_columns)

        self.constant_columns_count = len(self.constant_columns)

        self.constant_values = {col: data[col].unique()[0] for col in self.constant_columns}

        # 04 - Numerical columns handling ==========================================

        self.numerical_columns = np.array([col for col in self.names if col not in self.nominal_columns and col not in self.constant_columns])

        self.numerical_mask = np.isin(self.names, self.numerical_columns)

        self.numerical_columns_count = len(self.numerical_columns)
        # 05 - Non-negative columns handling =======================================

        # non-negative columns, each colum that has minimum value greater than zero
        self.non_negative_columns = np.array([col for col in self.names if np.min(data[col]) >= 0])

        self.non_negative_mask = np.isin(self.names, self.non_negative_columns)

        # 06 - Range values handling ===============================================

        # range values for numerical columns are the max - min
        self.range_values = np.array([(np.max(data[col]) - np.min(data[col])) for col in self.names])
        self.range_values = self.range_values.astype('float64')
        # the range value for nominal columns is 1
        self.range_values[self.nominal_mask] = 1.0

        # 07 standard deviation values handling ====================================

        # we should have a vector such that each cell of the vector
        # should contain the standard deviation of the corresponding column
        self.std_values = np.array([data[col].std() for col in self.names])





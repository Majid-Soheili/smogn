## load dependencies - third party
import numpy as np
import pandas as pd
import random as rd

from six import print_
from tqdm import tqdm

## load dependencies - internal
from smogn.box_plot_stats import box_plot_stats
from smogn.dist_metrics import euclidean_dist, heom_dist, overlap_dist


## generate synthetic observations
def over_sampling(

        ## arguments / inputs
        data,  ## training set
        index,  ## index of input data
        perc,  ## over / under sampling
        pert,  ## perturbation / noise percentage
        k,  ## num of neighs for over-sampling
        missing_values_thr = 0.1,  ## how to handle missing values
        seed=None,  ## random seed for sampling (pos int or None)
        verbose=True  ## print statements
):
    """
    generates synthetic observations and is the primary function underlying the
    over-sampling technique utilized in the higher main function 'smogn()', the
    4 step procedure for generating synthetic observations is:
    
    1) pre-processing: temporarily removes features without variation, label 
    encodes nominal / categorical features, and subsets the training set into 
    two data sets by data type: numeric / continuous, and nominal / categorical
    
    2) distances: calculates the cartesian distances between all observations, 
    distance metric automatically determined by data type (euclidean distance 
    for numeric only data, heom distance for both numeric and nominal data, and 
    hamming distance for nominal only data) and determine k nearest neighbors
    
    3) over-sampling: selects between two techniques, either synthetic minority 
    over-sampling technique for regression 'smoter' or 'smoter-gn' which applies
    a similar interpolation method to 'smoter', but perterbs the interpolated 
    values
    
    'smoter' is selected when the distance between a given observation and a 
    selected nearest neighbor is within the maximum threshold (half the median 
    distance of k nearest neighbors) 'smoter-gn' is selected when a given 
    observation and a selected nearest neighbor exceeds that same threshold
    
    both 'smoter' and 'smoter-gn' only applies to numeric / continuous features, 
    for nominal / categorical features, synthetic values are generated at random 
    from sampling observed values found within the same feature
    
    4) post processing: restores original values for label encoded features, 
    reintroduces constant features previously removed, converts any interpolated
    negative values to zero in the case of non-negative features
    
    returns a pandas dataframe containing synthetic observations of the training
    set which are then returned to the higher main function 'smogn()'
    
    ref:
    
    Branco, P., Torgo, L., Ribeiro, R. (2017).
    SMOGN: A Pre-Processing Approach for Imbalanced Regression.
    Proceedings of Machine Learning Research, 74:36-50.
    http://proceedings.mlr.press/v74/branco17a/branco17a.pdf.
    
    Branco, P., Ribeiro, R., Torgo, L. (2017). 
    Package 'UBL'. The Comprehensive R Archive Network (CRAN).
    https://cran.r-project.org/web/packages/UBL/UBL.pdf.
    """

    ## subset original dataframe by bump classification index
    data = data.iloc[index]

    ## store dimensions of data subset
    n = len(data)
    d = len(data.columns)

    ## store original data types
    feat_dtypes_orig = [None] * d

    for j in range(d):
        feat_dtypes_orig[j] = data.iloc[:, j].dtype

    ## find non-negative numeric features
    feat_non_neg = []
    num_dtypes = ["int64", "float64"]

    for j in range(d):
        if data.iloc[:, j].dtype in num_dtypes and all(data.iloc[:, j] > 0):
            feat_non_neg.append(j)

    ## find features without variation (constant features)
    feat_const = data.columns[data.nunique() == 1]

    ## temporarily remove constant features
    if len(feat_const) > 0:
        ## create copy of original data and omit constant features
        data_orig = data.copy()
        data = data.drop(columns=feat_const)

        ## store list of features with variation
        feat_var = list(data.columns.values)

        ## reindex features with variation
        data.reset_index(drop=True, inplace=True)
        data.columns = range(len(data.columns))

        ## store new dimension of feature space
        d = len(data.columns)

    ## create copy of data containing variation
    data_var = data.copy()

    ## create global feature list by column index
    feat_list = list(data.columns.values)

    ## create nominal feature list and
    ## label encode nominal / categorical features
    ## (strictly label encode, not one hot encode)

    mapping_dict = {}
    feat_list_nom = []
    nom_dtypes = ["object", "bool", "datetime64", "category"]
    for j in range(d):
        if data.dtypes[j] in nom_dtypes:
            feat_list_nom.append(j)

            if data.dtypes[j] == 'category':
                uniques = data.iloc[:, j].cat.categories
                data.iloc[:, j] = data.iloc[:, j].cat.codes.values.copy()
                #data.iloc[:, j] = pd.Categorical(data.iloc[:, j].cat.codes)
            else:
                codes, uniques = pd.factorize(data.iloc[:, j])
                data.iloc[:, j] = pd.Categorical(codes)

            mapping_dict[j] = dict(zip(range(len(uniques)), uniques))

    data = data.apply(pd.to_numeric)

    ## create numeric feature list
    feat_list_num = list(set(feat_list) - set(feat_list_nom))

    ## calculate ranges for numeric / continuous features
    ## (includes label encoded features)
    feat_ranges = list(np.repeat(1, d))
    if len(feat_list_nom) > 0:
        for j in feat_list_num:
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])
    else:
       for j in range(d):
            feat_ranges[j] = max(data.iloc[:, j]) - min(data.iloc[:, j])

    ## subset feature ranges to include only numeric features
    ## (excludes label encoded features)
    feat_ranges_num = np.array([feat_ranges[i] for i in feat_list_num])

    ## subset data by either numeric / continuous or nominal / categorical
    data_num = data.iloc[:, feat_list_num]
    data_nom = data.iloc[:, feat_list_nom]

    ## get number of features for each data type
    feat_count_num = len(feat_list_num)
    feat_count_nom = len(feat_list_nom)

    n_samples = n

    # Initialize the distance matrix
    dist_matrix = np.zeros((n_samples, n_samples))

    # Convert data to NumPy arrays for efficient computation
    data_num_array = None
    data_nom_array = None
    if feat_count_num > 0:
        data_num_array = data_num.values  # Shape: (n_samples, n_numeric_features)
    if feat_count_nom > 0:
        data_nom_array = data_nom.values  # Shape: (n_samples, n_nominal_features)


    # Ensure data_num_array does not contain NaNs
    if np.isnan(data_num_array).any():
        raise ValueError("Numeric data contains NaNs. Please handle missing values before proceeding.")

    if np.isnan(data_nom_array).any():
        raise ValueError("Nominal data contains NaNs. Please handle missing values before proceeding.")

    if np.isnan(feat_ranges_num).any():
        raise ValueError("Ranges of numeric features contain NaNs. Please handle missing values before proceeding.")

    if 0 in feat_ranges_num:
        raise ValueError("Numeric features contain zero range. Please remove constant features before proceeding.")


    # Compute the distance matrix
    if feat_count_num > 0 and feat_count_nom == 0:
        # Case 1: All features are numeric
        # Compute Euclidean distance using vectorized operations
        s = np.sum(data_num_array ** 2, axis=1)
        dist_matrix = np.sqrt(
            s[:, np.newaxis] + s[np.newaxis, :] - 2 * np.dot(data_num_array, data_num_array.T)
        )

    elif feat_count_nom > 0 and feat_count_num == 0:
        # Case 2: All features are nominal
        # Compute Hamming distance using vectorized operations
        diff_nom = data_nom_array[:, np.newaxis, :] != data_nom_array[np.newaxis, :, :]
        dist_matrix = np.sum(diff_nom, axis=2).astype(float)

    elif feat_count_num > 0 and feat_count_nom > 0:
        # Case 3: Mixed features (both numeric and nominal)
        # Numeric part
        diff_num = (
                    data_num_array[:, np.newaxis, :] - data_num_array[np.newaxis, :, :]
                   ) / feat_ranges_num  # Normalize differences

        diff_num **= 2  # Square differences
        sum_diff_num = np.sum(diff_num, axis=2)

        # Nominal part
        diff_nom = data_nom_array[:, np.newaxis, :] != data_nom_array[np.newaxis, :, :]
        diff_nom = diff_nom.astype(float)
        sum_diff_nom = np.sum(diff_nom, axis=2)

        # Combine numeric and nominal distances
        dist_matrix = np.sqrt(sum_diff_num + sum_diff_nom)

    else:
        # No features present
        dist_matrix = np.zeros((n_samples, n_samples))

    # Ensure the distance matrix does not contain NaNs
    if np.isnan(dist_matrix).any():
        raise ValueError("Distance matrix contains NaNs.")


    ## calculate distance between observations based on data types
    ## store results over null distance matrix of n x n
    #dist_matrix = np.zeros(shape=(n, n))
    #if feat_count_nom > 0:
#
    #    diff = data.values[:, np.newaxis, :] - data.values[np.newaxis, :, :]
    #    # Divide the differences by 'ranges_num'
    #    diff /= feat_ranges
#
    #    # For nominal indices, replace all non-zero values with 1
    #    diff[:, :, feat_list_num] = np.where(diff[:,:, feat_list_num] != 0, 1, 0)
#
    #    # Square the differences
    #    diff **= 2
#
    #    # Sum the squared differences along the last axis
    #    dist_num = np.sum(diff, axis=-1)
#
    #    # Take the square root of the sum
    #    dist_num = np.sqrt(dist_num)
    #    dist_matrix = dist_num
#
#    for i in tqdm(range(n), ascii=True, desc="dist_matrix"):
    #    for j in range(i + 1, n):
#
    #        ## utilize euclidean distance given that
    #        ## data is all numeric / continuous
    #        if feat_count_nom == 0:
    #            dist_matrix[i][j] = euclidean_dist(
    #                a=data_num.iloc[i],
    #                b=data_num.iloc[j],
    #                d=feat_count_num
    #            )#

    #        ## utilize heom distance given that
    #        ## data contains both numeric / continuous
    #        ## and nominal / categorical
    #        if feat_count_nom > 0 and feat_count_num > 0:#

    #            diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    #            # Divide the differences by 'ranges_num'
    #            diff /= feat_ranges
#
                # For nominal indices, replace all non-zero values with 1
    #            diff[:, :, feat_list_num] = np.where(diff[:,:, feat_list_num] != 0, 1, 0)

                # Square the differences
#                diff **= 2
#
     #           # Sum the squared differences along the last axis
     #           dist_num = np.sum(diff, axis=-1)
#
                # Take the square root of the sum
     #           dist_num = np.sqrt(dist_num)
 #               dist_matrix = dist_num
      #          dist_matrix[i][j] = heom_dist(
#
 ##                   ## numeric inputs
 #                   a_num=data_num.iloc[i],
      #              b_num=data_num.iloc[j],
  #    #              d_num=feat_count_num,
 #                   ranges_num=feat_ranges_num,
#
  #                  ## nominal inputs
       #             a_nom=data_nom.iloc[i],
   #                 b_nom=data_nom.iloc[j],
    #                d_nom=feat_count_nom
#                )
#
            ## utilize hamming distance given that 
            ## data is all nominal / categorical
#            if feat_count_num == 0:
#                dist_matrix[i][j] = overlap_dist(
#                    a=data_nom.iloc[i],
#                    b=data_nom.iloc[j],
#                    d=feat_count_nom
#                )
#            dist_matrix[j][i] = dist_matrix[i][j]

    ## determine indicies of k nearest neighbors
    ## and convert knn index list to matrix
    knn_index = [None] * n

    for i in range(n):
        knn_index[i] = np.argsort(dist_matrix[i])[1:k + 1]

    knn_matrix = np.array(knn_index)

    ## calculate max distances to determine if gaussian noise is applied
    ## (half the median of the distances per observation)
    max_dist = [None] * n

    for i in range(n):
        max_dist[i] = box_plot_stats(dist_matrix[i])["stats"][2] / 2

    ## number of new synthetic observations for each rare observation
    x_synth = int(perc - 1)

    ## total number of new synthetic observations to generate
    n_synth = int(n * (perc - 1 - x_synth))

    ## set random seed 
    if seed:
        np.random.seed(seed=seed)

    ## randomly index data by the number of new synthetic observations
    r_index = np.random.choice(
        a=tuple(range(0, n)),
        size=n_synth,
        replace=False,
        p=None
    )

    ## Quality Check the Standard Deviation of the features
    for x in range(d):
        scale = np.std(data.iloc[:, x].dropna())
        if np.isnan(scale):
            print(f"Standard deviation for feature {x} is NaN.")
            scale = 0  # or a small positive value


    ## create null matrix to store new synthetic observations
    synth_matrix = np.ndarray(shape=((x_synth * n + n_synth), d))

    if x_synth > 0:
        for i in tqdm(range(n), ascii=True, desc="synth_matrix", disable=not verbose):

            ## determine which cases are 'safe' to interpolate
            safe_list = np.where(
                dist_matrix[i, knn_matrix[i]] < max_dist[i])[0]

            for j in range(x_synth):

                ## set random seed 
                if seed:
                    np.random.seed(seed=seed)

                ## randomly select a k nearest neighbor
                neigh = int(np.random.choice(
                    a=tuple(range(k)),
                    size=1))

                ## conduct synthetic minority over-sampling
                ## technique for regression (smoter)
                if neigh in safe_list:
                    ## set random seed
                    if seed:
                        rd.seed(a=seed)

                    diffs = data.iloc[
                            knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[
                                                               i, 0:(d - 1)]
                    synth_matrix[i * x_synth + j, 0:(d - 1)] = data.iloc[
                                                               i, 0:(d - 1)] + rd.random() * diffs

                    ## randomly assign nominal / categorical features from
                    ## observed cases and selected neighbors
                    for x in feat_list_nom:
                        ## set random seed
                        if seed:
                            rd.seed(a=seed)

                        synth_matrix[i * x_synth + j, x] = [data.iloc[
                                                                knn_matrix[i, neigh], x], data.iloc[
                                                                i, x]][round(rd.random())]

                    ## generate synthetic y response variable by
                    ## inverse distance weighted
                    for z in feat_list_num[0:(d - 1)]:
                        a = abs(data.iloc[i, z] - synth_matrix[
                            i * x_synth + j, z]) / feat_ranges[z]
                        b = abs(data.iloc[knn_matrix[
                            i, neigh], z] - synth_matrix[
                                    i * x_synth + j, z]) / feat_ranges[z]

                    if len(feat_list_nom) > 0:
                        a = a + sum(data.iloc[
                                        i, feat_list_nom] != synth_matrix[
                                        i * x_synth + j, feat_list_nom])
                        b = b + sum(data.iloc[knn_matrix[
                            i, neigh], feat_list_nom] != synth_matrix[
                                        i * x_synth + j, feat_list_nom])

                    if a == b:
                        synth_matrix[i * x_synth + j,
                        (d - 1)] = (data.iloc[i, (d - 1)] + data.iloc[
                            knn_matrix[i, neigh], (d - 1)]) / 2
                    else:
                        synth_matrix[i * x_synth + j,
                        (d - 1)] = (b * data.iloc[
                            i, (d - 1)] + a * data.iloc[
                                        knn_matrix[i, neigh], (d - 1)]) / (a + b)

                ## conduct synthetic minority over-sampling technique
                ## for regression with the introduction of gaussian 
                ## noise (smoter-gn)
                else:
                    if max_dist[i] > pert:
                        t_pert = pert
                    else:
                        t_pert = max_dist[i]

                    index_gaus = i * x_synth + j

                    for x in range(d):
                        if pd.isna(data.iloc[i, x]):
                            synth_matrix[index_gaus, x] = None
                        else:
                            ## set random seed 
                            if seed:
                                np.random.seed(seed=seed)

                            synth_matrix[index_gaus, x] = data.iloc[
                                                              i, x] + float(np.random.normal(
                                loc=0,
                                scale=np.std(data.iloc[:, x]),
                                size=1) * t_pert)

                            if x in feat_list_nom:
                                unique_values = data.iloc[:, x].unique()
                                num_unique = len(unique_values)
                                if num_unique == 1:
                                    # Assign the sole unique value to the synth_matrix
                                    synth_matrix[index_gaus, x] = data.iloc[0, x]
                                else:

                                    # Calculate the frequency of each unique value using value_counts
                                    value_counts = data.iloc[:, x].value_counts().reindex(unique_values).fillna(0)

                                    # Convert counts to weights
                                    probs = value_counts.tolist()

                                    # Set random seed if provided
                                    if seed is not None:
                                        rd.seed(seed)

                                    # Select one value based on the computed probabilities
                                    selected_value = rd.choices(
                                        population=unique_values,
                                        weights=probs,
                                        k=1
                                    )[0]  # Extract the single value from the list

                                    # Assign the selected value to the synth_matrix
                                    synth_matrix[index_gaus, x] = selected_value

    if n_synth > 0:
        count = 0

        for i in tqdm(r_index, ascii=True, desc="r_index", disable=not verbose):

            ## determine which cases are 'safe' to interpolate
            safe_list = np.where(
                dist_matrix[i, knn_matrix[i]] < max_dist[i])[0]

            ## set random seed 
            if seed:
                np.random.seed(seed=seed)

            ## randomly select a k nearest neighbor
            neigh = int(np.random.choice(
                a=tuple(range(0, k)),
                size=1))

            ## conduct synthetic minority over-sampling 
            ## technique for regression (smoter)
            if neigh in safe_list:
                ##  set random seed
                if seed:
                    rd.seed(a=seed)

                diffs = data.iloc[
                        knn_matrix[i, neigh], 0:(d - 1)] - data.iloc[i, 0:(d - 1)]
                synth_matrix[x_synth * n + count, 0:(d - 1)] = data.iloc[
                                                               i, 0:(d - 1)] + rd.random() * diffs

                ## randomly assign nominal / categorical features from
                ## observed cases and selected neighbors
                for x in feat_list_nom:
                    ## set random seed
                    if seed:
                        rd.seed(a=seed)

                    synth_matrix[x_synth * n + count, x] = [data.iloc[
                                                                knn_matrix[i, neigh], x], data.iloc[
                                                                i, x]][round(rd.random())]

                ## generate synthetic y response variable by
                ## inverse distance weighted
                for z in feat_list_num[0:(d - 1)]:
                    a = abs(data.iloc[i, z] - synth_matrix[
                        x_synth * n + count, z]) / feat_ranges[z]
                    b = abs(data.iloc[knn_matrix[i, neigh], z] - synth_matrix[
                        x_synth * n + count, z]) / feat_ranges[z]

                if len(feat_list_nom) > 0:
                    a = a + sum(data.iloc[i, feat_list_nom] != synth_matrix[
                        x_synth * n + count, feat_list_nom])
                    b = b + sum(data.iloc[
                                    knn_matrix[i, neigh], feat_list_nom] != synth_matrix[
                                    x_synth * n + count, feat_list_nom])

                if a == b:
                    synth_matrix[x_synth * n + count, (d - 1)] = (data.iloc[
                                                                     i, (d - 1)] + data.iloc[
                                                                     knn_matrix[i, neigh], (d - 1)]) / 2
                else:
                    synth_matrix[x_synth * n + count, (d - 1)] = (b * data.iloc[
                        i, (d - 1)] + a * data.iloc[
                                                                      knn_matrix[i, neigh], (d - 1)]) / (a + b)

            ## conduct synthetic minority over-sampling technique
            ## for regression with the introduction of gaussian 
            ## noise (smoter-gn)
            else:
                if max_dist[i] > pert:
                    t_pert = pert
                else:
                    t_pert = max_dist[i]

                for x in range(d):
                    if pd.isna(data.iloc[i, x]):
                        synth_matrix[x_synth * n + count, x] = None
                    else:
                        ## set random seed 
                        if seed:
                            np.random.seed(seed=seed)

                        synth_matrix[x_synth * n + count, x] = data.iloc[
                                                                   i, x] + float(np.random.normal(
                            loc=0,
                            scale=np.std(data.iloc[:, x]),
                            size=1) * t_pert)

                        if x in feat_list_nom:
                            if len(data.iloc[:, x].unique()) == 1:
                                synth_matrix[
                                    x_synth * n + count, x] = data.iloc[0, x]
                            else:
                                # Extract unique values and their counts using pandas
                                unique_values = data.iloc[:, x].unique()
                                value_counts = data.iloc[:, x].value_counts().reindex(unique_values).fillna(0)

                                # Convert counts to probabilities (weights)
                                probs = value_counts.tolist()

                                # Set random seed if provided
                                if seed is not None:
                                    rd.seed(seed)

                                # Select one value based on the computed probabilities
                                selected_value = rd.choices(
                                    population=unique_values,
                                    weights=probs,
                                    k=1
                                )[0]  # Extract the single value from the list

                                # Assign the selected value to the synth_matrix
                                synth_matrix[x_synth * n + count, x] = selected_value
            ## close loop counter
            count = count + 1

    ## convert synthetic matrix to dataframe
    data_new = pd.DataFrame(synth_matrix)

    ## synthetic data quality check
    if sum(data_new.isnull().sum()) > 0:

        #compute how percentage of rows contain missing values
        n1 = len(data_new)
        n2 = data_new.isnull().any(axis=1).sum()
        missing_rows_percent = n2 / n1
        print_(f"Synthetic data contains missing values in {missing_rows_percent:.2%} of rows: {n2}/{n1}")
        #print_(f"Synthetic data contains missing values in {missing_rows_percent:.2%} of rows")
        if missing_rows_percent > missing_values_thr:
            raise ValueError(f"oops! synthetic data contains missing values in {missing_rows_percent:.2%} of rows")
        else:
            data_new = data_new.dropna()


    ## replace label encoded values with original values
    for j in feat_list_nom:
        #code_list = data.iloc[:, j].unique()
        #cat_list = data_var.iloc[:, j].unique()

        #for x in code_list:
        #    data_new.iloc[:, j] = data_new.iloc[:, j].replace(x, cat_list[x])
        mapping = mapping_dict[j]
        data_new.iloc[:, j] = data_new.iloc[:, j].map(mapping)

    ## reintroduce constant features previously removed
    if len(feat_const) > 0:
        data_new.columns = feat_var

        for j in range(len(feat_const)):
            data_new.insert(
                loc=int(feat_const[j]),
                column=feat_const[j],
                value=np.repeat(
                    data_orig.iloc[0, feat_const[j]],
                    len(synth_matrix))
            )

    ## convert negative values to zero in non-negative features
    for j in feat_non_neg:
        # data_new.iloc[:, j][data_new.iloc[:, j] < 0] = 0
        data_new.iloc[:, j] = data_new.iloc[:, j].clip(lower=0)

    ## return over-sampling results dataframe
    return data_new

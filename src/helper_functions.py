import glob
import pandas as pd


def combine_data(file_path):


    # Read and merge all CSV files with a 'Year' column
    dataframes = []
    for file in glob.glob(file_path):
        df = pd.read_csv(file)
        year = file.split("_")[-1].split(".")[0]  # Extract year from filename (format "statistics_table_2013.csv")
        df["Year"] = int(year)
        dataframes.append(df)   


    # Combine all data
    return pd.concat(dataframes)


def get_decoded_data_and_analyse(data, column_dict, predictors, outcomes):

    all_columns = predictors + outcomes
 
    print("Data values counts:")
    for i in all_columns:
        print(data[i].value_counts())

    print("\nData types of each column:")
    print(data[all_columns].dtypes)
    
    data_selected = data[all_columns].copy()
    # remove decode keyword from column name
    data_selected.columns = [col.replace("_decode", "") for col in data_selected.columns]

    data_selected.rename(columns=column_dict, inplace=True)

    return data_selected



def preprocess_data(categorical_columns, binary_columns, data_selected, column_dict):
    categorical_column_after = [] # after changing the column name to a readable one using the column_dict dictionary
    for i in categorical_columns:
        categorical_column_after.append(column_dict[i])


    binary_columns_after = [] # after changing the column name to a readable one using the column_dict dictionary
    for i in binary_columns:
        binary_columns_after.append(column_dict[i])


    for col in binary_columns_after:    # after changing the column name to a readable one
        # Direct column assignment (dtype-inferring) is required for pandas >= 3.0;
        # the prior `.loc[:, col] = ... .map(...)` form raised TypeError on pandas 3.x
        # because the source column is dtype='object'/str and the target values are int.
        data_selected[col] = data_selected[col].map({'Yes': 1, 'No': 0}).astype('Int64')


    # Collect all dummies first, then do ONE concat at the end. pandas 3.x on
    # Apple Silicon can segfault when a wide frame with mixed Int64 / object
    # dtypes is repeatedly concatenated in a Python loop; a single batched
    # concat avoids the repeated reallocation path.
    _dummies_list = []
    for col in categorical_column_after:
        dummies = pd.get_dummies(data_selected[col], prefix=col).astype(int)
        _dummies_list.append(dummies)

    if _dummies_list:
        data_selected = pd.concat([data_selected] + _dummies_list, axis=1, copy=False)

    return data_selected










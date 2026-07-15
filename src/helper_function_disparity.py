from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def disparity_with_agnostic(df_array, model_names, group_name = 'Gender', metric_col='Accuracy', subset_name='Overall'):
    """
    Plot a metric for a specific subset across multiple demographic groups.
    
    Parameters:
    - df: DataFrame with columns ['Subset', 'Group', 'Group Value', metric_col]
    - subset_name: The name of the subset to plot (e.g., 'Overall')
    - metric_col: The metric to plot on the Y-axis (e.g., 'Accuracy')
    """

    # Filter for the desired subset
    plot_dfs = []
    for i, df in enumerate(df_array):
        # FOR SEPERATE, ADD "DESCRIPTION" NAME "TRAINED_{distinct_value}_TESTED_{distinct_value}"
        if model_names[i] == 'Seperate LR':
            if "Group Value" in df.columns:
                temp_df = df.copy()
                # change the Age group values: change "65 or older" to "65Y or older" and "Years Old" to "Y"
                if group_name == 'Age':
                    temp_df['Group Value'] = (
                        temp_df['Group Value']
                        .str.replace(r'\s*Years Old', 'Y', regex=True)
                        .str.replace(r'^65 or Older$', '65Y or Older', regex=True)
                    )

                for distinct_value in temp_df["Group Value"].unique():
                    temp_df.loc[temp_df["Group Value"] == distinct_value, "Description"] = f"PLR: Train {distinct_value}, Test {distinct_value}"
                plot_dfs.append(temp_df)
          

        if model_names[i] == 'FAIR':
            if "Group Value" in df.columns:
                temp_df = df.copy()
                # change the Age group values: change "65 or older" to "65Y or older" and "Years Old" to "Y"
                if group_name == 'Age':
                    temp_df['Group Value'] = (
                        temp_df['Group Value']
                        .str.replace(r'\s*Years Old', 'Y', regex=True)
                        .str.replace(r'^65 or Older$', '65Y or Older', regex=True)
                    )

                # remove test on "All" from Group Value
                temp_df = temp_df[temp_df["Group Value"] != "All"]
                for distinct_value in temp_df["Group Value"].unique():
                    temp_df.loc[temp_df["Group Value"] == distinct_value, "Description"] = f"FAIR-PLR: Train All, Test {distinct_value}"
                plot_dfs.append(temp_df)

        if model_names[i] == 'Agnostic LR':
            if "Group Value" in df.columns:
                temp_df = df.copy()
                # change the Age group values: change "65 or older" to "65Y or older" and "Years Old" to "Y"
                if group_name == 'Age':
                    temp_df['Group Value'] = (
                        temp_df['Group Value']
                        .str.replace(r'\s*Years Old', 'Y', regex=True)
                        .str.replace(r'^65 or Older$', '65Y or Older', regex=True)
                    )

                temp_df = temp_df[temp_df["Group Value"] != "All"]
                for distinct_value in temp_df["Group Value"].unique():
                    temp_df.loc[temp_df["Group Value"] == distinct_value, "Description"] = f"Agnostic PLR: Train All, Test {distinct_value}"
                plot_dfs.append(temp_df)

    plot_df = pd.concat(plot_dfs, ignore_index=True)
    

   
    # if binary classification, replace '0' and '1' in the 'Group Value' column with 'No' and 'Yes'
    if group_name == "Drug Use Disorder (any past year)":
        value_map = {
            '0': 'No',
            '1': 'Yes'
            # Add more mappings as needed
        }
        
        plot_df['Group Value'] = plot_df['Group Value'].replace(value_map)
        # Replace '0' and '1' in the 'Description' column for 'Test' values
        for old, new in value_map.items():
            plot_df['Description'] = plot_df['Description'].str.replace(f'Test {old}', f'Test {new}')
        for old, new in value_map.items():
            plot_df['Description'] = plot_df['Description'].str.replace(f'Train {old}', f'Train {new}')
        
        
    

    # rounding the metric column if it is 'TP %' (12.123 -> 12.12)
    if metric_col == 'TP %':
        plot_df['TP %'] = plot_df['TP %'].round(2)
    
    # only consider the specific subset_name
    plot_df = plot_df[plot_df["Subset"] == subset_name]
    


    plot_df = plot_df.loc[:, ['Subset', 'Group', metric_col, 'Description']]

    print(plot_df)
    plot_df = plot_df.dropna(subset=['Description'])
    # Filter FAIR-PLR
    fair_df = plot_df[plot_df['Description'].str.startswith('FAIR-PLR')]
    fair_max = fair_df.loc[fair_df[metric_col].idxmax()]
    fair_min = fair_df.loc[fair_df[metric_col].idxmin()]

    fair_min["Type"]= "FAIR-PLR: Min"
    fair_max["Type"]= "FAIR-PLR: Max"


    # Filter PLR (but not FAIR-PLR)
    plr_df = plot_df[plot_df['Description'].str.startswith('PLR')]
    plr_max = plr_df.loc[plr_df[metric_col].idxmax()]
    plr_min = plr_df.loc[plr_df[metric_col].idxmin()]

    plr_min["Type"]= "PLR: Min"
    plr_max["Type"]= "PLR: Max"

    # filter Agnostic
    agnostic_df = plot_df[plot_df['Description'].str.startswith('Agnostic')]
    agnostic_max = agnostic_df.loc[agnostic_df[metric_col].idxmax()]
    agnostic_min = agnostic_df.loc[agnostic_df[metric_col].idxmin()]

    agnostic_min["Type"]= "Agnostic: Min"
    agnostic_max["Type"]= "Agnostic: Max"

    # Combine results into a new DataFrame
    result_df = pd.DataFrame([fair_max, fair_min, plr_max, plr_min, agnostic_max, agnostic_min])

    # Display the result
    print(result_df)
    result_df.to_csv(f"result/disparity/MaxMin_{group_name}_{metric_col}_{subset_name}.csv", index=False)



def disparity_without_agnostic(df_array, model_names, group_name = 'Gender', metric_col='Accuracy', subset_name='Overall'):
    """
    Plot a metric for a specific subset across multiple demographic groups.
    
    Parameters:
    - df: DataFrame with columns ['Subset', 'Group', 'Group Value', metric_col]
    - subset_name: The name of the subset to plot (e.g., 'Overall')
    - metric_col: The metric to plot on the Y-axis (e.g., 'Accuracy')
    """

    # Filter for the desired subset
    plot_dfs = []
    for i, df in enumerate(df_array):
        # FOR SEPERATE, ADD "DESCRIPTION" NAME "TRAINED_{distinct_value}_TESTED_{distinct_value}"
        if model_names[i] == 'Seperate LR':
            if "Group Value" in df.columns:
                temp_df = df.copy()
                # change the Age group values: change "65 or older" to "65Y or older" and "Years Old" to "Y"
                if group_name == 'Age':
                    temp_df['Group Value'] = (
                        temp_df['Group Value']
                        .str.replace(r'\s*Years Old', 'Y', regex=True)
                        .str.replace(r'^65 or Older$', '65Y or Older', regex=True)
                    )

                for distinct_value in temp_df["Group Value"].unique():
                    temp_df.loc[temp_df["Group Value"] == distinct_value, "Description"] = f"PLR: Train {distinct_value}, Test {distinct_value}"
                plot_dfs.append(temp_df)
          

        if model_names[i] == 'FAIR':
            if "Group Value" in df.columns:
                temp_df = df.copy()
                # change the Age group values: change "65 or older" to "65Y or older" and "Years Old" to "Y"
                if group_name == 'Age':
                    temp_df['Group Value'] = (
                        temp_df['Group Value']
                        .str.replace(r'\s*Years Old', 'Y', regex=True)
                        .str.replace(r'^65 or Older$', '65Y or Older', regex=True)
                    )

                # remove test on "All" from Group Value
                temp_df = temp_df[temp_df["Group Value"] != "All"]
                for distinct_value in temp_df["Group Value"].unique():
                    temp_df.loc[temp_df["Group Value"] == distinct_value, "Description"] = f"FAIR-PLR: Train All, Test {distinct_value}"
                plot_dfs.append(temp_df)

        
    plot_df = pd.concat(plot_dfs, ignore_index=True)
    

   
    # if binary classification, replace '0' and '1' in the 'Group Value' column with 'No' and 'Yes'
    if group_name == "Drug Use Disorder (any past year)":
        value_map = {
            '0': 'No',
            '1': 'Yes'
            # Add more mappings as needed
        }
        
        plot_df['Group Value'] = plot_df['Group Value'].replace(value_map)
        # Replace '0' and '1' in the 'Description' column for 'Test' values
        for old, new in value_map.items():
            plot_df['Description'] = plot_df['Description'].str.replace(f'Test {old}', f'Test {new}')
        for old, new in value_map.items():
            plot_df['Description'] = plot_df['Description'].str.replace(f'Train {old}', f'Train {new}')
        
        
    

    # rounding the metric column if it is 'TP %' (12.123 -> 12.12)
    if metric_col == 'TP %':
        plot_df['TP %'] = plot_df['TP %'].round(2)
    
    # only consider the specific subset_name
    plot_df = plot_df[plot_df["Subset"] == subset_name]
    


    plot_df = plot_df.loc[:, ['Subset', 'Group', metric_col, 'Description']]

    print(plot_df)
    plot_df = plot_df.dropna(subset=['Description'])
    # Filter FAIR-PLR
    fair_df = plot_df[plot_df['Description'].str.startswith('FAIR-PLR')]
    fair_max = fair_df.loc[fair_df[metric_col].idxmax()]
    fair_min = fair_df.loc[fair_df[metric_col].idxmin()]

    fair_min["Type"]= "FAIR-PLR: Min"
    fair_max["Type"]= "FAIR-PLR: Max"


    # Filter PLR (but not FAIR-PLR)
    plr_df = plot_df[plot_df['Description'].str.startswith('PLR')]
    plr_max = plr_df.loc[plr_df[metric_col].idxmax()]
    plr_min = plr_df.loc[plr_df[metric_col].idxmin()]

    plr_min["Type"]= "PLR: Min"
    plr_max["Type"]= "PLR: Max"

    
    # Combine results into a new DataFrame
    result_df = pd.DataFrame([fair_max, fair_min, plr_max, plr_min])

    # Display the result
    print(result_df)
    result_df.to_csv(f"result/disparity/MaxMin_{group_name}_{metric_col}_{subset_name}.csv", index=False)



   
   

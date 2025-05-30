import pandas as pd

def print_comparison_markdown(df_baseline, df_deepgemm, columns_to_display, sort_columns, new_column_name="Type"):
    """
    Processes two dataframes, adds a comparison column, sorts them,
    and prints them as Markdown tables.

    Args:
        df_baseline (pd.DataFrame): DataFrame for the baseline case (e.g., w/o deepgemm).
        df_deepgemm (pd.DataFrame): DataFrame for the comparison case (e.g., w deepgemm).
        columns_to_display (list): List of column names to include in the output table.
        sort_columns (list): List of column names to sort by.
        new_column_name (str): Name for the new column indicating the type.
    """
    if df_baseline is None or df_deepgemm is None:
        print("Error: One or both input DataFrames are None.")
        return

    # --- Process Baseline DataFrame ---
    try:
        df1_processed = df_baseline.copy()
        df1_processed[new_column_name] = 'w/o deepgemm'
        # Ensure all requested columns exist before selecting/sorting
        existing_cols_df1 = [col for col in columns_to_display if col in df1_processed.columns]
        sort_cols_df1 = [col for col in sort_columns if col in df1_processed.columns]
        if not sort_cols_df1: # If no sort columns exist, don't sort
             print("Warning: No sort columns found in baseline DataFrame. Skipping sort.")
             df1_processed = df1_processed[existing_cols_df1 + [new_column_name]]
        else:
             df1_processed = df1_processed[existing_cols_df1 + [new_column_name]]
             df1_processed = df1_processed.sort_values(by=sort_cols_df1)

        print("## Results for Baseline (w/o deepgemm)")
        print(df1_processed.to_markdown(index=False))
        print("\n") # Add newline for spacing

    except Exception as e:
        print(f"Error processing baseline DataFrame: {e}")


    # --- Process DeepGEMM DataFrame ---
    try:
        df2_processed = df_deepgemm.copy()
        df2_processed[new_column_name] = 'w deepgemm'
        # Ensure all requested columns exist before selecting/sorting
        existing_cols_df2 = [col for col in columns_to_display if col in df2_processed.columns]
        sort_cols_df2 = [col for col in sort_columns if col in df2_processed.columns]

        if not sort_cols_df2: # If no sort columns exist, don't sort
            print("Warning: No sort columns found in deepgemm DataFrame. Skipping sort.")
            df2_processed = df2_processed[existing_cols_df2 + [new_column_name]]
        else:
            df2_processed = df2_processed[existing_cols_df2 + [new_column_name]]
            df2_processed = df2_processed.sort_values(by=sort_cols_df2)


        print("## Results with DeepGEMM Enabled (w deepgemm)")
        print(df2_processed.to_markdown(index=False))
        print("\n") # Add newline for spacing

    except Exception as e:
        print(f"Error processing deepgemm DataFrame: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # In a real scenario, load df1_run4 and df2_run4 here
    # For example, using the function from your notebook:
    # from sglang_bench_analysis import load_benchmark_data_from_dir # Assuming you save that function
    # df1 = load_benchmark_data_from_dir('path/to/baseline/data')
    # df2 = load_benchmark_data_from_dir('path/to/deepgemm/data')
    # df1_run4 = df1[(df1['run'] == 4) & (df1['isl'] == 1024)] if df1 is not None else None
    # df2_run4 = df2[(df2['run'] == 4) & (df2['isl'] == 1024)] if df2 is not None else None

    # Create dummy data for demonstration if df1_run4/df2_run4 aren't loaded
    print("--- Running Example Usage with Dummy Data ---")
    data1 = {
        'n': [512, 512, 512, 512], 'isl': [1024, 1024, 1024, 1024], 'osl': [1, 1, 1, 1],
        'c': [8, 16, 32, 64], 'input_throughput': [6873, 8135, 8228, 8173],
        'mean_ttft_ms': [1179, 1986, 3890, 7551], 'other': [1]*4
    }
    df1_run4_dummy = pd.DataFrame(data1)

    data2 = {
        'n': [512, 512, 512, 512], 'isl': [1024, 1024, 1024, 1024], 'osl': [1, 1, 1, 1],
        'c': [8, 16, 32, 64], 'input_throughput': [8899, 10842, 11063, 10906],
        'mean_ttft_ms': [912, 1486, 2884, 5644], 'extra': [2]*4
    }
    df2_run4_dummy = pd.DataFrame(data2)

    # Define the columns to display and sort by
    cols_display = ['n', 'isl', 'osl', 'c', 'input_throughput', 'mean_ttft_ms']
    cols_sort = ['n', 'isl', 'osl', 'c']

    # Call the function with the dummy data
    print_comparison_markdown(df1_run4_dummy, df2_run4_dummy, cols_display, cols_sort)

    # If you were running this after loading real data, you would call:
    # print_comparison_markdown(df1_run4, df2_run4, cols_display, cols_sort) 
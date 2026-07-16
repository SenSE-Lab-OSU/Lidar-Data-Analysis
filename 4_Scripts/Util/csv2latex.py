#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def escape_latex_characters(df):
    """
    Escapes characters that break LaTeX compilation, specifically underscores 
    in string columns and column headers.
    """
    # Escape underscores in column names
    df.columns = [str(col).replace('_', '\\_') for col in df.columns]
    
    # Escape underscores in any string/object text data rows
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('_', '\\_', regex=False)
            
    return df

def interactive_column_selection(available_columns):
    """
    Prompts the user via terminal to select which columns to keep.
    """
    print("\nAvailable columns found in CSV:")
    for idx, col in enumerate(available_columns, start=1):
        print(f"  [{idx}] {col}")
        
    print("\nInstructions: Enter the numbers of the columns you want to keep (separated by commas).")
    print("Example: 1,3,4 (or press Enter to select ALL columns)")
    
    user_input = input("Selection: ").strip()
    if not user_input:
        print("-> Selecting all columns by default.")
        return list(available_columns)
        
    selected_cols = []
    try:
        indices = [int(i.strip()) - 1 for i in user_input.split(",") if i.strip()]
        for idx in indices:
            if 0 <= idx < len(available_columns):
                selected_cols.append(available_columns[idx])
            else:
                print(f"Warning: Index {idx+1} is out of bounds and will be skipped.")
    except ValueError:
        print("Error: Invalid numeric input format. Defaulting to all columns.")
        return list(available_columns)
        
    if not selected_cols:
        print("Warning: No valid columns selected. Defaulting to all columns.")
        return list(available_columns)
        
    print(f"-> Selected {len(selected_cols)} columns for export.")
    return selected_cols

def export_table(csv_path, format_type):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # 1. Load the CSV data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading file {csv_path}: {e}")
        return

    # 2. Interactive Column Selection
    selected_columns = interactive_column_selection(df.columns)
    df = df[selected_columns]

    # Clean up and transform header labels visually before processing
    df.columns = [col.replace('_', ' ').title() for col in df.columns]

    if format_type == 'aggregate':
        output_filename = f"{base_name}_summary_table.tex"
        print(f"Generating a compact summary table -> {output_filename}")
        
        # Determine grouping columns based on user choices
        group_candidates = ['Transform', 'Pair Label', 'Transform Method', 'Data Pair']
        group_cols = [c for c in df.columns if c in group_candidates]
        error_col = [c for c in df.columns if 'Error' in c or 'Err' in c]
        
        if group_cols and error_col:
            df_to_export = df.groupby(group_cols).agg(
                Total_Points=(df.columns[2] if len(df.columns) > 2 else df.columns[0], 'count'),
                Mean_Error=(error_col[0], 'mean'),
                Max_Error=(error_col[0], 'max')
            ).reset_index()
        else:
            print("Notice: Missing standard grouping keys for custom aggregation. Exporting raw slice.")
            df_to_export = df.copy()
            
        # Round float values
        float_cols = df_to_export.select_dtypes(include=['float64']).columns
        df_to_export[float_cols] = df_to_export[float_cols].round(2)
        
        # Generate the raw LaTeX text block from pandas
        raw_latex = df_to_export.to_latex(index=False)
        
        # --- DEFINITIVE ESCAPE FIX ---
        # Protect existing structure, escape actual data underscores, restore structure
        raw_latex = raw_latex.replace('\\', 'TEMPMARKERBACKSLASH')
        raw_latex = raw_latex.replace('_', '\\_')
        raw_latex = raw_latex.replace('TEMPMARKERBACKSLASH', '\\')
        
        latex_document = (
            "\\begin{table}[htbp]\n"
            f"\\caption{{Aggregated Metric Evaluation Log for {base_name.replace('_', ' ')}}}\n"
            f"\\label{{tab:{base_name}_summary}}\n"
            "\\centering\n"
            f"{raw_latex}"
            "\\end{table}\n"
        )
        
    else:  # Longtable format
        output_filename = f"{base_name}_long_table.tex"
        print(f"Generating a multi-page longtable -> {output_filename}")
        
        # Create a copy to match workflow structure
        df_to_export = df.copy()
        
        # Round numeric values
        float_cols = df_to_export.select_dtypes(include=['float64']).columns
        df_to_export[float_cols] = df_to_export[float_cols].round(2)
        
        # Generate the raw LaTeX text block from pandas
        raw_latex = df_to_export.to_latex(index=False)
        
        # --- DEFINITIVE ESCAPE FIX ---
        # Protect existing structure, escape actual data underscores, restore structure
        raw_latex = raw_latex.replace('\\', 'TEMPMARKERBACKSLASH')
        raw_latex = raw_latex.replace('_', '\\_')
        raw_latex = raw_latex.replace('TEMPMARKERBACKSLASH', '\\')
        
        # Convert to a clean multi-page longtable layout structure
        latex_document = raw_latex.replace('\\begin{tabular}', '\\begin{longtable}')
        latex_document = latex_document.replace('\\end{tabular}', '\\end{longtable}')
        
        header_insertion = (
            f"\\caption{{Exhaustive Dataset Tracking Matrix for {base_name.replace('_', ' ')}}} "
            f"\\label{{tab:{base_name}_exhaustive}} \\\\\n"
        )
        latex_document = latex_document.replace('\\toprule\n', f'\\toprule\n{header_insertion}')

    # 3. Write final text to the local file
    with open(output_filename, 'w') as f:
        f.write(latex_document)
    print(f"Export completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an engineering evaluation CSV file to an IEEE Transactions LaTeX table format with dynamic filtering."
    )
    parser.add_argument(
        "csv_path", 
        type=str, 
        help="Path to the input CSV data file (e.g., path/to/error.csv)"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--long", 
        action="store_true", 
        help="Generate a multi-page longtable block perfect for an appendix layout."
    )
    group.add_argument(
        "--aggregate", 
        action="store_true", 
        help="Generate a condensed, grouped data metric summary table for the main text."
    )
    
    args = parser.parse_args()
    chosen_format = 'long' if args.long else 'aggregate'
    
    export_table(args.csv_path, chosen_format)
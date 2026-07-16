import os
import re
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def natural_sort_key(folder_name):
    """Splits folder name into chunks of text and numbers for natural numeric sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', folder_name)]

def clean_numeric_column(series):
    """Strips trailing single quotes and converts back to true numeric types."""
    return pd.to_numeric(series.astype(str).str.replace("'", "", regex=False), errors='coerce')

def process_reprojection_data(base_directory):
    results = []
    found_relative_paths = []
    
    base_directory = os.path.abspath(base_directory)
    top_level_file = os.path.join(base_directory, "overall_reprojection_error.csv")
    
    # 1. Target determination
    if os.path.exists(top_level_file):
        print("[INFO] Found 'overall_reprojection_error.csv' at the top level. Bypassing subdirectory search.")
        targets = [(base_directory, ["overall_reprojection_error.csv"])]
    else:
        print("[INFO] Top-level file not found. Searching through subdirectories...")
        targets = os.walk(base_directory)

    # 2. Extract Data
    for root, dirs, files in targets:
        for file in files:
            if file == "overall_reprojection_error.csv":
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_directory)
                found_relative_paths.append(rel_path)
                
                folder_name = os.path.basename(root) if os.path.basename(root) != "" else "Root"
                
                try:
                    df = pd.read_csv(filepath)
                    
                    # Clean all columns required for metrics & plots
                    df['confidence'] = clean_numeric_column(df['confidence'])
                    df['quality'] = clean_numeric_column(df['quality'])
                    df['error_confidence'] = clean_numeric_column(df['error_confidence'])
                    df['error_px'] = clean_numeric_column(df['error_px'])
                    
                    df = df.dropna(subset=['confidence', 'quality', 'error_confidence', 'error_px'])
                    
                    # Compute Table Metrics
                    mean_conf = df['confidence'].mean()
                    mean_quality = df['quality'].mean()
                    
                    top_5_quality = df.sort_values(by='quality', ascending=False).head(5)
                    mean_err_conf_top5 = top_5_quality['error_confidence'].mean()
                    
                    # Compute Plot Metric
                    mean_error_px = df['error_px'].mean()
                    
                    if pd.isna(mean_conf) or pd.isna(mean_quality) or pd.isna(mean_err_conf_top5) or pd.isna(mean_error_px):
                        continue
                        
                    results.append({
                        'Folder': folder_name,
                        'Mean Confidence': mean_conf,
                        'Mean Quality': mean_quality,
                        'Top 5 Err. Conf.': mean_err_conf_top5,
                        'Mean Error (px)': mean_error_px
                    })
                except Exception as e:
                    print(f"Error processing file at {rel_path}: {e}")

    if not results:
        print("[ERROR] No valid data could be aggregated.")
        return

    # Convert to DataFrame to safely calculate ranks
    res_df = pd.DataFrame(results)

    # 3. JOINT OPTIMIZATION RANKING CALCULATION
    # Higher confidence is better -> ascending=False
    res_df['rank_conf'] = res_df['Mean Confidence'].rank(ascending=False)
    # Higher quality is better -> ascending=False
    res_df['rank_qual'] = res_df['Mean Quality'].rank(ascending=False)
    # Lower Top 5 error confidence is better -> ascending=True
    res_df['rank_err_conf'] = res_df['Top 5 Err. Conf.'].rank(ascending=True)

    # Joint score is the sum of ranks (lowest sum = closest to the top of all 3 metrics)
    res_df['joint_rank_score'] = res_df['rank_conf'] + res_df['rank_qual'] + res_df['rank_err_conf']
    
    # Extract the absolute best folder
    best_idx = res_df['joint_rank_score'].idxmin()
    best_folder = res_df.loc[best_idx, 'Folder']
    
    print("\n" + "="*60)
    print(f"BEST JOINT PERFORMANCE: {best_folder}")
    print(f"  • Mean Confidence:   {res_df.loc[best_idx, 'Mean Confidence']:.4f}")
    print(f"  • Mean Quality:      {res_df.loc[best_idx, 'Mean Quality']:.6f}")
    print(f"  • Top 5 Err. Conf.:  {res_df.loc[best_idx, 'Top 5 Err. Conf.']:.4f}")
    print("="*60 + "\n")

    # 4. Sort results naturally by Folder name for LaTeX table consistency
    sorted_results = sorted(results, key=lambda x: natural_sort_key(x['Folder']))
    
    # 5. Generate the 4-Column Metric / 3-Subtable Row-Interleaved LaTeX Format
    num_items = len(sorted_results)
    chunk_size = math.ceil(num_items / 3)
    
    col1 = sorted_results[0:chunk_size]
    col2 = sorted_results[chunk_size:2*chunk_size]
    col3 = sorted_results[2*chunk_size:]
    
    latex_str = "\\begin{table*}[t]\n\\centering\n"
    latex_str += "\\caption{Reprojection Analysis Summary Across Dataset Folders}\n"
    latex_str += "\\label{tab:reprojection_summary}\n"
    latex_str += "\\scriptsize\n"
    latex_str += "\\setlength{\\tabcolsep}{2pt}\n"
    latex_str += "\\begin{tabular}{lcccclcccclcccc}\n"
    latex_str += "\\hline\n"
    latex_str += "\\textbf{Folder} & \\textbf{Mean} & \\textbf{Mean} & \\textbf{Top 5} & & \\textbf{Folder} & \\textbf{Mean} & \\textbf{Mean} & \\textbf{Top 5} & & \\textbf{Folder} & \\textbf{Mean} & \\textbf{Mean} & \\textbf{Top 5} \\\\\n"
    latex_str += " & \\textbf{Conf.} & \\textbf{Qual.} & \\textbf{Err. Conf.} & & & \\textbf{Conf.} & \\textbf{Qual.} & \\textbf{Err. Conf.} & & & \\textbf{Conf.} & \\textbf{Qual.} & \\textbf{Err. Conf.} \\\\\n"
    latex_str += "\\hline\n"
    
    for i in range(chunk_size):
        row_str = "  "
        if i < len(col1):
            row_str += f"{col1[i]['Folder']} & {col1[i]['Mean Confidence']:.4f} & {col1[i]['Mean Quality']:.6f} & {col1[i]['Top 5 Err. Conf.']:.4f}"
        else:
            row_str += " & & & "
        row_str += " & & "
        if i < len(col2):
            row_str += f"{col2[i]['Folder']} & {col2[i]['Mean Confidence']:.4f} & {col2[i]['Mean Quality']:.6f} & {col2[i]['Top 5 Err. Conf.']:.4f}"
        else:
            row_str += " & & & "
        row_str += " & & "
        if i < len(col3):
            row_str += f"{col3[i]['Folder']} & {col3[i]['Mean Confidence']:.4f} & {col3[i]['Mean Quality']:.6f} & {col3[i]['Top 5 Err. Conf.']:.4f}"
        else:
            row_str += " & & & "
        row_str += " \\\\\n"
        latex_str += row_str
        
    latex_str += "\\hline\n\\end{tabular}\n\\end{table*}\n"
    
    output_txt_path = os.path.join(base_directory, "reprojection_analysis_table.txt")
    with open(output_txt_path, "w") as f:
        f.write(latex_str)
    print(f"[SUCCESS] Exported naturally sorted table with folder quality to: {output_txt_path}")
    
    # 6. Generate and save visualization plot
    plt.figure(figsize=(14, 6))
    bars = plt.bar(res_df['Folder'], res_df['Mean Error (px)'], color='#c0392b', edgecolor='black', alpha=0.85)
    plt.yscale('log')
    
    min_idx = res_df['Mean Error (px)'].idxmin()
    min_folder = res_df.loc[min_idx, 'Folder']
    min_value = res_df.loc[min_idx, 'Mean Error (px)']
    
    bars[min_idx].set_color('#27ae60')
    bars[min_idx].set_edgecolor('black')
    
    plt.annotate(
        f'Lowest Error:\n{min_folder} ({min_value:.2f} px)',
        xy=(min_idx, min_value),
        xytext=(min_idx, min_value * 2.5 if min_value > 0 else 5),
        textcoords='data',
        ha='center',
        va='bottom',
        color='#1e7e34',
        fontweight='bold',
        fontsize=9,
        arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6, headlength=6)
    )
    
    plt.title('Mean Reprojection Error per Folder from error_px (Log Scale)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Folder Name', fontsize=10, fontweight='bold')
    plt.ylabel('Mean Reprojection Error (px - Log)', fontsize=10, fontweight='bold')
    plt.xticks(rotation=90, fontsize=6)
    plt.grid(axis='y', which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    plot_output_path = os.path.join(base_directory, "mean_error_confidence_plot.png")
    plt.savefig(plot_output_path, dpi=300)
    print(f"[SUCCESS] Saved logarithmic pixel error plot to: {plot_output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process overall reprojection error csv files safely into multi-column LaTeX output.")
    parser.add_argument("DIR", type=str, help="The top-level target directory containing folders.")
    args = parser.parse_args()
    process_reprojection_data(args.DIR)
#!/usr/bin/env python3
"""Process tensorpow benchmark results.

Generates scientific paper quality plots and LaTeX tables from the
summary JSON output.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def format_memory(bytes_val):
    if bytes_val is None:
        return "---"
    mb = bytes_val / (1024**2)
    if mb > 100000:
        return f"{mb:.1e}"
    else:
        return f"{mb:.1f}"

def format_scientific(val):
    if val is None:
        return "---"
    if val < 0.001:
        return f"{val:.1e}"
    elif val < 0.1:
        return f"{val:.3f}"
    else:
        return f"{val:.2f}"

def theoretical_direct_bytes(n):
    # A single complex128 matrix of size 3^n x 3^n
    return 16 * (9 ** n)

def theoretical_max_block_bytes(n):
    # D(k) = dimension of k-th symmetric power of C^3
    def D(k):
        return (k + 1) * (k + 2) // 2

    max_block_dim = 0
    # Iterate over all valid integer partitions of n into at most 3 parts
    for l3 in range(n + 1):
        for l2_full in range(l3, n + 1):
            l1_full = n - l2_full - l3
            if l1_full < l2_full:
                continue
            
            l1 = l1_full - l2_full
            l2 = l2_full - l3
            
            # Block 1: Sym^{l1+l2} x Sym^{l2}
            dim1 = D(l1 + l2) * D(l2)
            max_block_dim = max(max_block_dim, dim1)
            
            # Block 2: Sym^{l1+l2+1} x Sym^{l2-1}
            if l2 >= 1:
                dim2 = D(l1 + l2 + 1) * D(l2 - 1)
                max_block_dim = max(max_block_dim, dim2)
                
    # A single complex128 matrix of size max_block_dim x max_block_dim
    return 16 * (max_block_dim ** 2)

def generate_latex_table(case_name, data, output_path):
    # Organize data by n
    table_data = defaultdict(dict)
    for row in data:
        n = row["n"]
        phase = row["phase"]
        method = row["method"]
        key = f"{method}_{phase}"
        
        table_data[n][key + "_time"] = row["median_seconds"]
            
        if "rel_residual" in row and row["rel_residual"] is not None:
            table_data[n]["error"] = row["rel_residual"]
            
    # Write LaTeX
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% \\usepackage{booktabs}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Benchmark results for the " + case_name.replace("_", " ") + " case.}\n")
        f.write("\\label{tab:benchmark_" + case_name + "}\n")
        f.write("\\begin{tabular}{r r r r r r r}\n")
        f.write("\\toprule\n")
        f.write("$n$ & Direct (s) & VB Warm (s) & VB Cold (s) & Direct Mem. (MB) & VB Max Block (MB) & Max Rel. Err. \\\\\n")
        f.write("\\midrule\n")
        
        for n in sorted(table_data.keys()):
            d = table_data[n]
            t_dir = format_scientific(d.get("direct_kronecker_warm_time"))
            t_warm = format_scientific(d.get("virtual_block_warm_time"))
            t_cold = format_scientific(d.get("virtual_block_cold_process_time"))
            
            mem_dir = format_memory(theoretical_direct_bytes(n))
            mem_vb = format_memory(theoretical_max_block_bytes(n))
            
            err = format_scientific(d.get("error"))
            
            f.write(f"{n} & {t_dir} & {t_warm} & {t_cold} & {mem_dir} & {mem_vb} & {err} \\\\\n")
            
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

def generate_plot(case_name, data, output_path):
    plt.style.use('seaborn-v0_8-paper')
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    
    # Extract series
    n_dir, t_dir = [], []
    n_warm, t_warm = [], []
    n_cold, t_cold = [], []
    
    for row in data:
        if row["method"] == "direct_kronecker" and row["phase"] == "warm":
            n_dir.append(row["n"])
            t_dir.append(row["median_seconds"])
        elif row["method"] == "virtual_block" and row["phase"] == "warm":
            n_warm.append(row["n"])
            t_warm.append(row["median_seconds"])
        elif row["method"] == "virtual_block" and row["phase"] == "cold_process":
            n_cold.append(row["n"])
            t_cold.append(row["median_seconds"])
            
    if n_dir:
        ax.plot(n_dir, t_dir, marker='s', linestyle='--', color='#d62728', label='Direct Kronecker')
    if n_warm:
        ax.plot(n_warm, t_warm, marker='o', linestyle='-', color='#1f77b4', label='Virtual Block (Warm)')
    if n_cold:
        ax.plot(n_cold, t_cold, marker='^', linestyle=':', color='#ff7f0e', label='Virtual Block (Cold)')
        
    ax.set_yscale('log')
    ax.set_xlabel('Tensor Power ($n$)', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    
    # Improve tick labels
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=10, loc='best')
    
    title = "Two-Term" if case_name == "two_term" else "Three-Term"
    ax.set_title(f"Scaling of Execution Time: {title} Case", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Process tensorpow benchmark results.")
    parser.add_argument("--input", default="results/benchmark_results_summary.json", help="Path to summary JSON")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return 1
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Group by case
    cases = defaultdict(list)
    for row in data:
        cases[row["case"]].append(row)
        
    output_dir = input_path.parent
    
    for case_name, case_data in cases.items():
        tex_path = output_dir / f"table_{case_name}.tex"
        plot_path = output_dir / f"plot_{case_name}.png"
        
        generate_latex_table(case_name, case_data, tex_path)
        generate_plot(case_name, case_data, plot_path)
        
        print(f"Generated {tex_path}")
        print(f"Generated {plot_path}")
        
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

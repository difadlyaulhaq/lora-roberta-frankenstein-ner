import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations(csv_path, output_dir="results/plots"):
    """
    Read the grid search results CSV and generate beautiful analytical plots for the thesis.
    """
    print(f"Reading grid search results from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})
    
    # ---------------------------------------------------------
    # Plot 1: Heatmap of Rank vs Alpha for F1-Score
    # ---------------------------------------------------------
    print("Generating Heatmap of Rank vs Alpha on F1-score...")
    pivot_f1 = df.pivot(index="rank", columns="alpha", values="f1")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_f1, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Macro F1-Score'})
    plt.title("Pengaruh Kombinasi Rank (r) dan Alpha (α) terhadap F1-Score")
    plt.xlabel("Alpha (α)")
    plt.ylabel("Rank (r)")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "heatmap_rank_alpha_f1.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 2: Bar Chart comparing MUC-5 Errors across combinations
    # ---------------------------------------------------------
    print("Generating MUC-5 Error Analysis bar chart...")
    # Create combination label: "r4_a16", "r4_a32", etc.
    df["label"] = df.apply(lambda row: f"r{int(row['rank'])}_a{int(row['alpha'])}", axis=1)
    
    muc5_cols = ["muc5_COR", "muc5_INC", "muc5_MIS", "muc5_SPU"]
    # Rename columns for Indonesian label presentation
    muc_df = df[["label"] + muc5_cols].copy()
    muc_df.columns = ["Model", "Correct (COR)", "Incorrect (INC)", "Missing (MIS)", "Spurious (SPU)"]
    
    # Melt dataframe for seaborn barplot
    muc_melted = pd.melt(muc_df, id_vars="Model", var_name="Kategori MUC-5", value_name="Jumlah Token")
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=muc_melted, x="Model", y="Jumlah Token", hue="Kategori MUC-5", palette="muted")
    plt.title("Analisis Kesalahan Berdasarkan Kriteria MUC-5")
    plt.xlabel("Kombinasi Hyperparameter LoRA")
    plt.ylabel("Jumlah Token Entitas")
    plt.xticks(rotation=45)
    plt.legend(title="Klasifikasi Galat MUC-5")
    plt.tight_layout()
    muc5_path = os.path.join(output_dir, "muc5_error_comparison.png")
    plt.savefig(muc5_path, dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 3: Line Chart comparing Fine-Grained Attributes
    # ---------------------------------------------------------
    print("Generating Fine-Grained attributes comparison chart...")
    # Compare eLen (Short vs Long) and eCon (Consistent vs Inconsistent) for best/representative runs
    # Let's plot accuracies of eLen_long_acc, eCon_inconsistent_acc, eFre_few_shot_acc
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(df["label"]))
    width = 0.25
    
    plt.bar(x - width, df["eLen_long_acc"], width, label="eLen Long (Frasa Panjang)", color="#3498db")
    plt.bar(x, df["eCon_inconsistent_acc"], width, label="eCon Inconsistent (Ambiguitas)", color="#e67e22")
    plt.bar(x + width, df["eFre_few_shot_acc"], width, label="eFre Few-Shot (Langka)", color="#2ecc71")
    
    plt.title("Performa Model pada Kelompok Atribut Entitas Non-Tipikal (Robustness)")
    plt.xlabel("Kombinasi Hyperparameter LoRA")
    plt.ylabel("Accuracy / Recall Rate")
    plt.xticks(x, df["label"], rotation=45)
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    fine_grained_path = os.path.join(output_dir, "fine_grained_robustness_comparison.png")
    plt.savefig(fine_grained_path, dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 4: Training Time comparison based on Rank and Alpha
    # ---------------------------------------------------------
    print("Generating Training Time comparison line graph...")
    plt.figure(figsize=(10, 6))
    
    # Plot training time line for different ranks grouped by alpha
    for alpha_val in sorted(df["alpha"].unique()):
        subset = df[df["alpha"] == alpha_val].sort_values("rank")
        plt.plot(subset["rank"], subset["training_time_sec"] / 3600.0, marker='o', linewidth=2, label=f"Alpha (α) = {alpha_val}")
        
    plt.title("Efisiensi Waktu Pelatihan Berdasarkan Rank dan Alpha")
    plt.xlabel("Rank (r)")
    plt.ylabel("Waktu Pelatihan (Jam)")
    plt.xticks(sorted(df["rank"].unique()))
    plt.legend(title="Skala Alpha")
    plt.tight_layout()
    time_path = os.path.join(output_dir, "training_time_comparison.png")
    plt.savefig(time_path, dpi=300)
    plt.close()

    print(f"\nSuccess! Generated 4 plots inside: {output_dir}")
    print(f"1. Heatmap F1-score: {heatmap_path}")
    print(f"2. Perbandingan MUC-5: {muc5_path}")
    print(f"3. Ketahanan Fine-Grained: {fine_grained_path}")
    print(f"4. Waktu Pelatihan: {time_path}")

if __name__ == "__main__":
    csv_file = "results/grid_search_results.csv"
    generate_visualizations(csv_file)

import uproot
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd


#This script plots and saves the residual of the momentum from QTracker_Prod and momentum_training. 
#This script also saves the momenta in a cvs, for plotting the comparison between models. 
#All plots are saved in /plots
def plot_and_save_png(data_mup, data_mum, xlabel, title, filepath):
    plt.figure(figsize=(10, 6))
    plt.hist(data_mup, bins=100, histtype='step', color='red', label='mu+')
    plt.hist(data_mum, bins=100, histtype='step', color='blue', label='mu−')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"[✓] Saved plot: {filepath}")

def main(filename: str, model_number: int):
    # Create 'plots' directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Load ROOT file and tree
    file = uproot.open(filename)
    tree = file["tree"]

    # Load branches
    gpx_array = tree["gpx"].arrays(library="np")["gpx"]
    gpy_array = tree["gpy"].arrays(library="np")["gpy"]
    gpz_array = tree["gpz"].arrays(library="np")["gpz"]

    qpx_array = tree["qpx"].arrays(library="np")["qpx"]
    qpy_array = tree["qpy"].arrays(library="np")["qpy"]
    qpz_array = tree["qpz"].arrays(library="np")["qpz"]

    # Compute residuals
    residuals = {
        'px_mup': [], 'px_mum': [],
        'py_mup': [], 'py_mum': [],
        'pz_mup': [], 'pz_mum': [],
    }

    for g, q in zip(gpx_array, qpx_array):
        if len(g) >= 2 and len(q) == 2:
            residuals['px_mup'].append(q[0] - g[0])
            residuals['px_mum'].append(q[1] - g[1])

    for g, q in zip(gpy_array, qpy_array):
        if len(g) >= 2 and len(q) == 2:
            residuals['py_mup'].append(q[0] - g[0])
            residuals['py_mum'].append(q[1] - g[1])

    for g, q in zip(gpz_array, qpz_array):
        if len(g) >= 2 and len(q) == 2:
            residuals['pz_mup'].append(q[0] - g[0])
            residuals['pz_mum'].append(q[1] - g[1])

    # Save plots
    plot_and_save_png(residuals['px_mup'], residuals['px_mum'],
                      xlabel="Residual(qpx - gpx) [GeV/c]",
                      title="Residual px (reco - true)",
                      filepath=f"plots/model_{model_number}_residual_px.png")

    plot_and_save_png(residuals['py_mup'], residuals['py_mum'],
                      xlabel="Residual(qpy - gpy) [GeV/c]",
                      title="Residual py (reco - true)",
                      filepath=f"plots/model_{model_number}_residual_py.png")

    plot_and_save_png(residuals['pz_mup'], residuals['pz_mum'],
                      xlabel="Residual(qpz - gpz) [GeV/c]",
                      title="Residual pz (reco - true)",
                      filepath=f"plots/model_{model_number}_residual_pz.png")
    
    # Compute mean and std
    stats = {
        'model_number': model_number,
        'px_mup_mean': np.mean(residuals['px_mup']),
        'px_mup_std': np.std(residuals['px_mup']),
        'px_mum_mean': np.mean(residuals['px_mum']),
        'px_mum_std': np.std(residuals['px_mum']),
        'py_mup_mean': np.mean(residuals['py_mup']),
        'py_mup_std': np.std(residuals['py_mup']),
        'py_mum_mean': np.mean(residuals['py_mum']),
        'py_mum_std': np.std(residuals['py_mum']),
        'pz_mup_mean': np.mean(residuals['pz_mup']),
        'pz_mup_std': np.std(residuals['pz_mup']),
        'pz_mum_mean': np.mean(residuals['pz_mum']),
        'pz_mum_std': np.std(residuals['pz_mum']),
    }

    # Check for model_residuals.csv
    db_path = "model_residuals.csv"
    if os.path.exists(db_path):
        df = pd.read_csv(db_path)
        if model_number in df["model_number"].values:
            print(f"[!] Model {model_number} already exists in database. Skipping stats save.")
        else:
            df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
    else:
        df = pd.DataFrame([stats])

    df.to_csv(db_path, index=False)
    print(f"[✓] Saved residual stats to {db_path}")

    if not os.path.exists(db_path):
        raise FileNotFoundError("model_residuals.csv does not exist. Run the residual analysis script first.")

    df = pd.read_csv(db_path)

    df = df.sort_values("model_number")

    # Setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)
    components = ['px', 'py', 'pz']
    colors = {'mup': 'tab:blue', 'mum': 'tab:red'}
    markers = {'mup': 'o', 'mum': '^'}

    # Plot each component (px, py, pz)
    for i, comp in enumerate(components):
        ax = axes[i]
        for particle in ['mup', 'mum']:
            means = df[f"{comp}_{particle}_mean"]
            stds = df[f"{comp}_{particle}_std"]
            model_ids = df["model_number"]

            ax.errorbar(
                model_ids,
                means,
                yerr=stds,
                fmt=markers[particle],
                color=colors[particle],
                label=f"{comp.upper()} {particle}",
                capsize=5
            )
        
        ax.set_title(f"{comp.upper()} Residuals by Model", fontsize=14)
        ax.set_xlabel("Model Number", fontsize=12)
        ax.set_ylabel("Residual Mean (GeV/c)", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=10)

    plt.tight_layout()

    # Save
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/model_residual_summary.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to ROOT file")
    parser.add_argument("model_number", type=int, help="Model number to tag output")
    args = parser.parse_args()

    main(args.filename, args.model_number)

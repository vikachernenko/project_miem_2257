import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
except ImportError:
    print("WARNING: 'seaborn' library not found. Using default matplotlib style.")
    print("To fix: pip install seaborn")
    plt.style.use('ggplot')

plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, choices=[1,2], required=True, help="Dimension: 1 or 2")
parser.add_argument("--dir", type=str, default=".", help="Directory with CSV files")
args = parser.parse_args()

dim = args.dim
folder = args.dir

if dim == 1:
    sigma_file = os.path.join(folder, "sigma_1d.csv")
    hist_file = os.path.join(folder, "histogram_1d.csv")
    paths_file = os.path.join(folder, "sample_paths_1d.csv")
    title_suffix = "1D Simulation"
else:
    sigma_file = os.path.join(folder, "sigma_2d.csv")
    hist_file = os.path.join(folder, "histogram_2d.csv")
    paths_file = os.path.join(folder, "sample_paths_2d.csv")
    title_suffix = "2D Simulation"

# 1. Plot Sigma vs Steps
if os.path.exists(sigma_file):
    sigma = pd.read_csv(sigma_file)
    step_col = "step" if "step" in sigma.columns else "steps"
    
    plt.figure(figsize=(10, 6))
    plt.plot(sigma[step_col], sigma["sigma_empirical"], 
             lw=3, color="crimson", label="Empirical (Simulation)", alpha=0.8)
    
    theory_label = fr"Theory ($\sqrt{{{'N' if dim==1 else '2N'}}}$)"
    plt.plot(sigma[step_col], sigma["sigma_theory"], 
             '--', color="navy", lw=2, label=theory_label)
    
    plt.xlabel("Number of Steps (N)", fontsize=14)
    plt.ylabel(r"$\sigma$ (RMS Displacement)", fontsize=14)
    plt.title(fr"Diffusion Law: $\sigma$ vs Steps ({title_suffix})", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"plot_{dim}d_sigma.png", dpi=300)
    plt.show()

# 2. Plot Histogram
if os.path.exists(hist_file):
    hist = pd.read_csv(hist_file)
    plt.figure(figsize=(10, 6))
    
    if dim == 1:
        plt.bar(hist["position"], hist["count"], width=1.0, 
                color="skyblue", alpha=0.7, label="Simulation")
        
        try:
            from scipy.stats import norm
            if os.path.exists(sigma_file):
                last_sigma = sigma["sigma_empirical"].iloc[-1]
                
                total_counts = hist["count"].sum()
                
                x = np.linspace(hist["position"].min(), hist["position"].max(), 200)
                p = norm.pdf(x, 0, last_sigma) * total_counts
                
                plt.plot(x, p, 'k--', linewidth=2, label='Normal Distribution Fit')
        except ImportError:
            pass

        plt.xlabel("Position", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.title("Final Position Distribution (1D)", fontsize=16)
        
    else:
        centers = (hist["r_min"] + hist["r_max"]) / 2
        width = hist["r_max"] - hist["r_min"]
        plt.bar(centers, hist["count"], width=width, color="purple", alpha=0.6, label="Simulation")
        
        plt.xlabel(r"Radius ($r = \sqrt{x^2+y^2}$)", fontsize=14)
        plt.ylabel("Count of Walkers", fontsize=14)
        plt.title("Radial Distribution (Distance from origin)", fontsize=16)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_{dim}d_hist.png", dpi=300)
    plt.show()


# 3. Plot Sample Paths (Trajectories)

if os.path.exists(paths_file):
    paths = pd.read_csv(paths_file)
    
    if dim == 1:
        plt.figure(figsize=(12, 6))
        unique_walkers = paths["walker"].unique()

        colors = plt.cm.jet(np.linspace(0, 1, len(unique_walkers)))
        
        for i, wid in enumerate(unique_walkers):
            g = paths[paths["walker"] == wid]
            plt.plot(g["step"], g["position"], lw=1.5, alpha=0.8, color=colors[i])
            
        plt.xlabel("Step Time", fontsize=14)
        plt.ylabel("Position", fontsize=14)
        plt.title(f"Sample Random Walks (1D)", fontsize=16)
        plt.tight_layout()
        plt.savefig("plot_1d_paths.png", dpi=300)
        plt.show()
        
    else:
        plt.figure(figsize=(8, 8))
        unique_walkers = paths["walker"].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_walkers)))

        plt.plot(0, 0, 'rx', markersize=10, markeredgewidth=2, label="Start", zorder=10)
        
        for i, wid in enumerate(unique_walkers):
            g = paths[paths["walker"] == wid]
            plt.plot(g["x"], g["y"], lw=1.5, alpha=0.7, color=colors[i], label=f"Walker {wid}")
            plt.plot(g["x"].iloc[-1], g["y"].iloc[-1], 'o', color=colors[i], markersize=5)

        plt.axis("equal")
        plt.xlabel("X Coordinate", fontsize=14)
        plt.ylabel("Y Coordinate", fontsize=14)
        plt.title(f"Sample 2D Random Walks Trajectories", fontsize=16)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("plot_2d_paths.png", dpi=300)
        plt.show()

else:
    print(f"No sample paths file found: {paths_file}")
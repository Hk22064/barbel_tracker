import numpy as np
import matplotlib.pyplot as plt

def plot_velocity(npy_path, label, ax):
    if not npy_path: return
    
    data = np.load(npy_path)
    ax.plot(data, label=f"{label} (Max: {np.max(data):.2f})")
    ax.axhline(0.05, color='r', linestyle='--', alpha=0.3, label="Thresh (0.05)")
    ax.axhline(-0.05, color='r', linestyle='--', alpha=0.3)
    
    # Calculate noise floor (approximated by median absolute deviation relative to 0)
    # or just mean abs value
    mean_abs = np.mean(np.abs(data))
    ax.set_title(f"{label} - MeanAbsVel: {mean_abs:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
    
    try:
        plot_velocity("output/debug_mybench.npy", "Success (mybench)", ax1)
    except FileNotFoundError:
        print("debug_mybench.npy not found")
        
    try:
        plot_velocity("output/debug_fail.npy", "Fail (front_9rep)", ax2)
    except FileNotFoundError:
        print("debug_fail.npy not found")
        
    plt.tight_layout()
    plt.savefig("output/debug_comparison_plot.png")
    print("Saved comparison plot to output/debug_comparison_plot.png")

if __name__ == "__main__":
    main()

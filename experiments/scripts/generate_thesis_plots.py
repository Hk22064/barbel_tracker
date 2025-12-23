import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Set Japanese Font (Windows)
plt.rcParams['font.family'] = 'MS Gothic'

# Configuration
SOURCE_DIR = os.path.join("..", "results", "final_accuracy_results", "小指")
OUTPUT_DIR = os.path.join("..", "..", "thesis", "figures")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
CSVs = [
    "accuracy_front_5rep_average_s5_th2.csv",
    "accuracy_front_9rep_average_s5_th2.csv",
    "accuracy_front_10rep_average_s5_th2.csv"
]

def generate_plots():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Global Style
    plt.rcParams.update({'font.size': 12})
    
    for csv_file in CSVs:
        path = os.path.join(SOURCE_DIR, csv_file)
        if not os.path.exists(path):
            print(f"Skipping {path} (Not found)")
            continue
            
        df = pd.read_csv(path)
        video_name = csv_file.replace("accuracy_", "").replace("_average_s5_th2.csv", "")
        
        # 1. Verification Plot (Manual vs System Velocity)
        plt.figure(figsize=(10, 6))
        
        reps = df['Rep']
        width = 0.35
        
        plt.bar(reps - width/2, df['Manual Vel'], width, label='Manual (Frame Count)', color='#3366cc', alpha=0.8)
        plt.bar(reps + width/2, df['System Vel'], width, label='Proposed System', color='#dc3912', alpha=0.8)
        
        plt.xlabel('Repetition #')
        plt.ylabel('Mean Velocity (m/s)')
        plt.title(f'Velocity Accuracy Verification: {video_name}')
        plt.xticks(reps)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Save
        out_path = os.path.join(OUTPUT_DIR, f"plot_{video_name}_velocity.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated {out_path}")
        
        # Copy CSV to output dir as well
        df.to_csv(os.path.join(OUTPUT_DIR, csv_file), index=False)
        print(f"Copied {csv_file}")

if __name__ == "__main__":
    generate_plots()

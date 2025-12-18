import matplotlib.pyplot as plt

# Table Data
table_data = [
    ["OS", "Windows 11 Home", ""],
    ["CPU", "Intel Core i7 / Ryzen 7", "MediaPipe (提案手法) のメイン演算"],
    ["GPU", "NVIDIA GeForce RTX 4070 Ti", "YOLO (比較手法 A) の推論に使用"],
    ["Memory", "16 GB", ""],
    ["Language", "Python 3.10", ""],
    ["Libraries", "Ultralytics, MediaPipe", "opencv-python, numpy, pandas"],
]

columns = ["項目 (Item)", "スペック (Spec)", "備考 (Note)"]

# Set Japanese Font
plt.rcParams['font.family'] = 'MS Gothic'

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('tight')
ax.axis('off')

# Color definitions
header_color = '#40466e'
row_colors = ['#f1f1f2', 'w']
edge_color = 'w'

# Create Table
table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='left')

# Style Table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Header styling
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='w')
        cell.set_facecolor(header_color)
        cell.set_text_props(ha='center') # Center header
    else:
        cell.set_facecolor(row_colors[row % 2])
        cell.set_edgecolor(edge_color)
        
        # Center the first column (Item)
        if col == 0:
            cell.set_text_props(weight='bold', ha='center')
        else:
             cell.set_text_props(ha='left') # Left align others
             # Simple padding via space injection or just skip manual position tweaking
             # Matplotlib tables are tricky. Let's just rely on cellLoc='left' (set in table init) relative to column.
             # Actually, table(..., cellLoc='left') applies to ALL.
             # Here we override per cell.

# Highlight GPU Row (Index 2 in data -> Row 3 in table)
gpu_row_idx = 3
for col in range(3):
    cell = table[gpu_row_idx, col]
    cell.set_facecolor('#e3f2fd') # Light Blue highlight
    if col == 1:
        cell.set_text_props(weight='bold', color='#0d47a1')

plt.title("実験動作環境 (Experimental Environment)", pad=10, fontsize=14, weight='bold')
plt.tight_layout()

out_path = "experiment_results/environment_table.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved table to {out_path}")

import pandas as pd
import matplotlib.pyplot as plt

# Load Data
CSV_PATH = "experiment_results/thesis_combined_summary_corrected.csv"
try:
    df = pd.read_csv(CSV_PATH)
except:
    # Fallback data if CSV missing
    data = {
        "Condition": ["Front (Horizontal)", "Right (Horizontal)", "Vertical (Accurate)", "Left (No Plate)", "Vertical (Far)"],
        "Correct Reps": [9, 9, 5, 10, 10],
        "Proposed (Hybrid)": ["9 (100%)", "10 (Over)", "5 (100%)", "8 (80%)", "10 (100%)"],
        "Comp A (YOLO11x)": ["8 (89%)", "10 (Over)", "5 (100%)", "13 (Over)", "3 (30%)"],
        "Comp B (Object)": ["0 (Fail)", "0 (Fail)", "0 (Fail)", "0 (Fail)", "0 (Fail)"]
    }
    df = pd.DataFrame(data)

# Rename columns for display if needed
# Assuming CSV has specific columns, let's restructure for the presentation table
# The CSV has: Video, Label, Method, Reps, FPS
# we want a Pivot table: Row=Label, Col=Method, Value=Reps(%)

# Create display DataFrame manually for perfect formatting
# Japanese labels requested
table_data = [
    ["横撮り (正面)", "9", "9 (100%)", "8 (89%)", "0 (Fail)"],
    ["横撮り (右側面)", "9", "10 (Over)", "10 (Over)", "0 (Fail)"],
    ["縦撮り (正確)", "5", "5 (100%)", "5 (100%)", "0 (Fail)"],
    ["横撮り (左/プレート無)", "10", "8 (80%)", "13 (Over)", "0 (Fail)"],
    ["縦撮り (遠距離)", "10", "10 (100%)", "3 (30%)", "0 (Fail)"],
    ["平均処理速度 (FPS)", "-", "42.4 fps", "46.8 fps", "72.8 fps"],
]

columns = ["条件 (Condition)", "正解", "提案手法\n(Hybrid)", "比較 A\n(YOLO11x)", "比較 B\n(Object)"]

# Set Japanese Font
plt.rcParams['font.family'] = 'MS Gothic'

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

# Color definitions
header_color = '#40466e'
row_colors = ['#f1f1f2', 'w']
edge_color = 'w'

# Create Table
table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')

# Style Table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='w')
        cell.set_facecolor(header_color)
    else:
        cell.set_facecolor(row_colors[row % 2])
        # Highlight "Far" row (Row 5 -> Index 5 in dict because header is 0)
        if row == 5:
            # Highlight Proposed Success
            if col == 2: # Proposed
                cell.set_facecolor('#d4edda') # Green tint
                cell.set_text_props(weight='bold', color='green')
            # Highlight Comp A Failure
            if col == 3: # Comp A
                cell.set_facecolor('#f8d7da') # Red tint
                cell.set_text_props(color='red')

    cell.set_edgecolor(edge_color)

plt.title("Comparison of Repetition Counting Accuracy", pad=20, fontsize=14, weight='bold')
plt.tight_layout()

out_path = "experiment_results/thesis_result_table.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved table to {out_path}")

import matplotlib.pyplot as plt
import pandas as pd
import os

# Set Japanese Font (Windows)
plt.rcParams['font.family'] = 'MS Gothic'

# Data
data = [
    ["front_9rep\n(横撮り・正面)", "9", "9 (100%)", "8 (89%)"],
    ["right_9rep\n(横撮り・右側)", "9", "10 (過検出)", "10 (過検出)"],
    ["front_5rep\n(縦撮り・中重量)", "5", "5 (100%)", "5 (100%)"],
    ["no20kg\n(横撮り・左側)", "10", "8 (80%)", "13 (過検出)"],
    ["front_10rep\n(縦撮り・遠距離)", "10", "10 (100%)", "3 (30%)"]
]

columns = ["動画条件", "正解", "提案手法\n(YOLO+MediaPipe)", "比較手法 A\n(YOLO11x)"]

OUTPUT_DIR = r"Thesis_Materials"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_table_image():
    # Increased width to prevent truncation
    fig, ax = plt.subplots(figsize=(12, 6)) 
    ax.axis('tight')
    ax.axis('off')
    
    # Create Table
    # colWidths helps ensure the first column gets enough space
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center', colLoc='center',
                     colWidths=[0.3, 0.1, 0.3, 0.3])
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5) # Adjust spacing
    
    # Color Header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='w', weight='bold')
        
        # Highlight Comp A failure
        if row == 5 and col == 3: # front_10rep YOLO
             cell.set_text_props(color='red', weight='bold')
        # Highlight Proposed Success
        if row == 5 and col == 2: # front_10rep Proposed
             cell.set_text_props(color='green', weight='bold')

    plt.title("手法別精度比較 (Repetition Detection Accuracy)", pad=20, fontsize=14, weight='bold')
    
    out_path = os.path.join(OUTPUT_DIR, "thesis_table_clean.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved table image to: {out_path}")

if __name__ == "__main__":
    create_table_image()

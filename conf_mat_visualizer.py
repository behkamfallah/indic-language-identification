"""Visualization script for Confusion Matrix JSON.

Usage:
    python conf_mat_visualizer.py --file path/to/confusion_matrix.json --output path/to/confusion_matrix.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_confusion_matrix(matrix, labels, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Confusion Matrix from JSON")
    parser.add_argument("--file", type=Path, required=True, help="Path to confusion_matrix.json")
    parser.add_argument("--output", type=Path, default=None, help="Path to save the output image (default: same dir as input)")
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: File {args.file} not found.")
        return

    with open(args.file, 'r') as f:
        data = json.load(f)
        
    matrix = np.array(data['matrix'])
    labels = data['labels']
    
    output_path = args.output
    if output_path is None:
        output_path = args.file.parent / "confusion_matrix.png"
        
    plot_confusion_matrix(matrix, labels, output_path)

if __name__ == "__main__":
    main()

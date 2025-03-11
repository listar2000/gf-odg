import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from dataloader import extract_clean_colors

BASE_OUTPUT_DIR = "/home/jiaweizhang/gf-odg/inference_results_gemma2"

def plot_color_histogram(colors, output_path):
    """
    Plots a histogram showing the frequency of colors mapped to numbers 1-6 and saves the image.
    """
    counts = Counter(colors)  # Count occurrences of each color
    labels = np.arange(1, 7)  # Labels for colors mapped to 1-6
    values = [counts[i] for i in labels]  # Get counts for 1-6

    color_names = ['Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Purple']
    bar_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']

    plt.figure(figsize=(8, 5))
    plt.bar(color_names, values, color=bar_colors)
    
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', fontsize=12)

    plt.xlabel("Colors")
    plt.ylabel("Frequency")
    plt.title("Histogram of Extracted Colors")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(output_path)  # Save the figure
    plt.close()  # Prevent memory leak in repeated runs


# Set the input CSV path
input_csv = os.path.join(BASE_OUTPUT_DIR, "inference_finetuned9e-5kl_0.0000001_color_3.csv")
input_csv_base = os.path.join(BASE_OUTPUT_DIR, "inference_base_color_3.csv")

input_csv_6 = os.path.join(BASE_OUTPUT_DIR, "inference_finetuned9e-5kl_0.0000001_color.csv")
input_csv_base_6 = os.path.join(BASE_OUTPUT_DIR, "inference_base_color.csv")

# Get cleaned colors (flatten list of lists)
cleaned_colors = extract_clean_colors(input_csv, N=3)
cleaned_colors_base = extract_clean_colors(input_csv_base, N=3)

cleaned_colors_6 = extract_clean_colors(input_csv_6, N=6)
cleaned_colors_base_6 = extract_clean_colors(input_csv_base_6, N=6)

print(cleaned_colors[:2])  # Print first few color lists for verification
flattened_colors = [color for sublist in cleaned_colors for color in sublist]  # Convert list of lists to a single list
flattened_colors_base = [color for sublist in cleaned_colors_base for color in sublist]  # Convert list of lists to a single list
flattened_colors_6 = [color for sublist in cleaned_colors_6 for color in sublist]  # Convert list of lists to a single list
flattened_colors_base_6 = [color for sublist in cleaned_colors_base_6 for color in sublist]  # Convert list of lists to a single list


print(flattened_colors[:10])  # Print first 10 colors for verification

# Define output file path
output_histogram_path = os.path.join(BASE_OUTPUT_DIR, "color_histogram_finetuned_color_3.png")
output_histogram_path_base = os.path.join(BASE_OUTPUT_DIR, "color_histogram_base_color_3.png")
output_histogram_path_6 = os.path.join(BASE_OUTPUT_DIR, "color_histogram_finetuned_color_6.png")
output_histogram_path_base_6 = os.path.join(BASE_OUTPUT_DIR, "color_histogram_base_color_6.png")

# Plot the histogram
plot_color_histogram(flattened_colors, output_histogram_path)
plot_color_histogram(flattened_colors_base, output_histogram_path_base)
plot_color_histogram(flattened_colors_6, output_histogram_path_6)
plot_color_histogram(flattened_colors_base_6, output_histogram_path_base_6)


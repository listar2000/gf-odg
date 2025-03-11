import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from dataloader import extract_clean_numbers,extract_clean_colors


BASE_OUTPUT_DIR = "/home/jiaweizhang/gf-odg/inference_results_gemma2"

def plot_histogram(numbers, output_path, N=6):
    """
    Plots a histogram showing the frequency of numbers 1-N and saves the image.

    Parameters:
    - numbers: List of numbers to plot.
    - output_path: Path to save the histogram image.
    - N: Upper bound of the number range (default is 6).
    """
    counts = Counter(numbers)  # Count occurrences of each number
    labels = np.arange(1, N + 1)  # Labels for 1 to N
    values = [counts[i] for i in labels]  # Get counts for 1 to N

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=plt.colormaps.get_cmap("tab10").colors[:N])  
    
    for i, v in enumerate(values):
        plt.text(i + 1, v + 0.5, str(v), ha='center', fontsize=12)

    plt.xlabel(f"Numbers (1-{N})")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Generated Numbers (1-{N})")
    plt.xticks(labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)  # Save the figure
    plt.close()  # Prevent memory leak in repeated runs

# Set the input CSV path
#input_csv = os.path.join(BASE_OUTPUT_DIR, "inference_base.csv")
input_csv = os.path.join(BASE_OUTPUT_DIR, "inference_finetuned9e-5kl_0.0000001_3_1to6.csv")
input_csv_base = os.path.join(BASE_OUTPUT_DIR, "inference_base_3_1to6.csv")
# Get cleaned numbers (flatten list of lists)
cleaned_numbers = extract_clean_numbers(input_csv, N=3)
cleaned_numbers_base = extract_clean_numbers(input_csv_base, N=3)

print(cleaned_numbers[:2])  # Print first
flattened_numbers = [num for sublist in cleaned_numbers for num in sublist]  # Convert list of lists to a single list
flattened_numbers_base = [num for sublist in cleaned_numbers_base for num in sublist]  # Convert list of lists to a single list
print(flattened_numbers[:10])  # Print first 10 numbers for verification

# Define output file path
#output_histogram_path = os.path.join(BASE_OUTPUT_DIR, "histogram_base.png")
output_histogram_path = os.path.join(BASE_OUTPUT_DIR, "histogram_finetuned_3.png")
output_histogram_path_base = os.path.join(BASE_OUTPUT_DIR, "histogram_base_3.png")


# Plot the histogram
plot_histogram(flattened_numbers, output_histogram_path)
plot_histogram(flattened_numbers_base, output_histogram_path_base)




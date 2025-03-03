import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
from dataloader import extract_clean_numbers

BASE_OUTPUT_DIR = "/home/jiaweizhang/gf-odg/inference_results"

def plot_histogram(numbers, output_path):
    """
    Plots a histogram showing the frequency of numbers 1-6 and saves the image.
    """
    counts = Counter(numbers)  # Count occurrences of each number
    labels = np.arange(1, 7)  # Labels for 1-6
    values = [counts[i] for i in labels]  # Get counts for 1-6

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
    
    for i, v in enumerate(values):
        plt.text(i + 1, v + 0.5, str(v), ha='center', fontsize=12)

    plt.xlabel("Numbers (1-6)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Generated Numbers (1-6)")
    plt.xticks(labels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)  # Save the figure
    plt.close()  # Prevent memory leak in repeated runs

# Set the input CSV path
#input_csv = os.path.join(BASE_OUTPUT_DIR, "inference_base.csv")
input_csv = os.path.join(BASE_OUTPUT_DIR, "inference_finetuned9e-5kl_0.0000001.csv")

# Get cleaned numbers (flatten list of lists)
cleaned_numbers = extract_clean_numbers(input_csv)
print(cleaned_numbers[:2])  # Print first
flattened_numbers = [num for sublist in cleaned_numbers for num in sublist]  # Convert list of lists to a single list

print(flattened_numbers[:10])  # Print first 10 numbers for verification

# Define output file path
#output_histogram_path = os.path.join(BASE_OUTPUT_DIR, "histogram_base.png")
output_histogram_path = os.path.join(BASE_OUTPUT_DIR, "histogram_finetuned.png")

# Plot the histogram
plot_histogram(flattened_numbers, output_histogram_path)

print(f"✅ Histogram saved to {output_histogram_path}")

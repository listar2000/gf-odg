import pandas as pd

def extract_clean_numbers(file_path):
    """
    Reads a CSV file using pandas, removes rows where 'Extracted Numbers' is empty 
    or does not contain exactly six numbers, and returns a list of lists of numbers.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Drop rows where 'Extracted Numbers' is NaN (empty)
    df = df.dropna(subset=['Extracted Numbers'])

    # Split 'Extracted Numbers' into lists
    df['Extracted Numbers'] = df['Extracted Numbers'].apply(lambda x: str(x).split(","))

    # Filter out rows that do not have exactly six numbers
    df = df[df['Extracted Numbers'].apply(lambda x: len(x) == 6)]

    # Convert numbers to integers
    clean_numbers = df['Extracted Numbers'].apply(lambda x: [int(num) for num in x]).tolist()

    return clean_numbers  # Return the list of lists of numbers

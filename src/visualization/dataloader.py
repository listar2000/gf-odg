import pandas as pd

def extract_clean_numbers(file_path, N=6):
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
    df = df[df['Extracted Numbers'].apply(lambda x: len(x) == N)]

    # Convert numbers to integers
    clean_numbers = df['Extracted Numbers'].apply(lambda x: [int(num) for num in x]).tolist()

    return clean_numbers  # Return the list of lists of numbers

# Define a mapping of colors to numbers
COLOR_MAPPING = {
    'Red': 1,
    'Blue': 2,
    'Green': 3,
    'Yellow': 4,
    'Orange': 5,
    'Purple': 6
}

def extract_clean_colors(file_path, N=6):
    """
    Reads a CSV file, extracts and cleans colors from 'Extracted Numbers',
    and maps them to numbers.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Drop rows where 'Extracted Numbers' is NaN (empty)
    df = df.dropna(subset=['Extracted Numbers'])

    # Split 'Extracted Numbers' into lists
    df['Extracted Numbers'] = df['Extracted Numbers'].apply(lambda x: str(x).split(","))

    # Filter out rows that do not have exactly N colors
    df = df[df['Extracted Numbers'].apply(lambda x: len(x) == N)]

    # Map colors to numbers
    clean_colors = df['Extracted Numbers'].apply(lambda x: [COLOR_MAPPING.get(color.strip(), None) for color in x]).tolist()

    return clean_colors  # Return the list of lists of mapped color numbers

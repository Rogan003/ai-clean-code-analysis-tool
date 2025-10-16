import csv
import os
import math

def split_csv(input_file, num_files=200):
    """Split a CSV file into multiple smaller files."""

    # Read the input file to count rows
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    total_rows = len(rows)
    rows_per_file = math.ceil(total_rows / num_files)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(input_file), 'split')
    os.makedirs(output_dir, exist_ok=True)

    # Split the data
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)

        if start_idx >= total_rows:
            break

        output_file = os.path.join(output_dir, f'd4j_extracted_part_{i+1:03d}.csv')

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows[start_idx:end_idx])

        print(f"Created {output_file} with {end_idx - start_idx} rows")

    print(f"\nTotal: {total_rows} rows split into {min(num_files, math.ceil(total_rows / rows_per_file))} files")

if __name__ == '__main__':
    input_csv = '../data/processed/d4j_extracted.csv'
    split_csv(input_csv, num_files=200)

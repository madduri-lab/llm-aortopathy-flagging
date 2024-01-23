"""
This script generates small preview files for the csv files in 
the MIMIC datasets by only extracting the first ten rows.
"""

import os
import csv

def save_first_ten_rows(input_file, output_file):
    try:
        with open(input_file, 'r', newline='') as csv_input_file:
            with open(output_file, 'w', newline='') as csv_output_file:
                reader = csv.reader(csv_input_file)
                writer = csv.writer(csv_output_file)

                # Read and write the header (if exists)
                header = next(reader)
                writer.writerow(header)

                # Read and write the first ten rows
                for _ in range(10):
                    try:
                        row = next(reader)
                        writer.writerow(row)
                    except StopIteration:
                        break

        print(f"First ten rows from {input_file} saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

DATA_DIRECTORY = "physionet.org/files/mimic-iv-note/2.2/note"
FILES = ['discharge', 'discharge_detail', 'radiology', 'radiology_detail']

for file in FILES:
    save_first_ten_rows(
        input_file=os.path.join(DATA_DIRECTORY, f"{file}.csv"),
        output_file=os.path.join(DATA_DIRECTORY, f"{file}_preview.csv")
    )


#This script is secondary, it adds the expected testing recommendation (based on ground truth), 
#and provides the proportion of the replicates for each note inference that either recommended 
#or did not recommend testing

import csv
import argparse
from collections import defaultdict, Counter

def aggregate_data(files):
    # Initialize a dictionary to store results for each patient
    results = defaultdict(lambda: {'tests': [], 'label': None})

    # Loop through each file and aggregate data
    for file_path in files:
        with open(file_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                results[row['ID']]['tests'].append(row['Testing'])
                results[row['ID']]['label'] = row['Label']

    return results

def calculate_concordance(results):
    # Initialize a dictionary to store concordance data
    concordance_data = {}

    # Calculate concordance for each patient
    for patient_id, data in results.items():
        test_counter = Counter(data['tests'])
        total_tests = len(data['tests'])
        concordance = {test: count / total_tests for test, count in test_counter.items()}
        # Determine expected outcome based on the label
        expected = "not_recommended" if data['label'] == "none" else "recommended"
        concordance_data[patient_id] = {'label': data['label'], 'expected': expected, 'testing': concordance}

    return concordance_data

def write_summary(concordance_data, output_file):
    # Define the header based on unique test outcomes
    outcomes = set(outcome for data in concordance_data.values() for outcome in data['testing'])
    header = ['ID', 'Label', 'Expected'] + sorted(outcomes)

    with open(output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        
        for patient_id, data in concordance_data.items():
            row = [patient_id, data['label'], data['expected']] + [data['testing'].get(outcome, 0) for outcome in sorted(outcomes)]
            csv_writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Aggregate testing results across multiple files and calculate concordance, including labels and expected outcomes.')
    parser.add_argument('output_file', type=str, help='Path to the output summary CSV file')
    parser.add_argument('input_files', nargs='+', help='Paths to the input CSV files')
    
    args = parser.parse_args()

    # Aggregate results from all input files
    aggregated_results = aggregate_data(args.input_files)
    
    # Calculate concordance
    concordance_data = calculate_concordance(aggregated_results)
    
    # Write summary to the output file
    write_summary(concordance_data, args.output_file)

if __name__ == "__main__":
    main()
#This script parses through a databricks output from this package to extract 
#the patient ID, input disease label, and the output testing recommendation

import re
import csv
import argparse

def parse_llm_output(input_file, output_file):
    data = []
    with open(input_file, 'r') as file:
        # Compile the regular expressions for extracting information
        patient_info_pattern = re.compile(r'- INFO - The label for the patient (\d+) is (\w+),')
        testing_pattern = re.compile(r'"testing":"([^"]+)"')
        
        # Initialize variables to keep track of data
        patient_id = None
        label = None
        
        for line in file:
            if '- INFO - The label' in line:
                patient_info_match = patient_info_pattern.search(line)
                if patient_info_match:
                    patient_id = int(patient_info_match.group(1))  # Convert ID to integer for sorting
                    label = patient_info_match.group(2)
            elif 'patient' in line and patient_id and label:
                testing_match = testing_pattern.search(line)
                if testing_match:
                    testing = testing_match.group(1)
                    # Store data in a list of dictionaries
                    data.append({'ID': patient_id, 'Label': label, 'Testing': testing})
                    patient_id, label = None, None  # Reset for the next record

    # Sort the data by ID
    sorted_data = sorted(data, key=lambda x: x['ID'])

    # Write sorted data to CSV
    with open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['ID', 'Label', 'Testing'])
        for entry in sorted_data:
            csv_writer.writerow([entry['ID'], entry['Label'], entry['Testing']])

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Parse LLM output and extract specified data to CSV, sorted by patient ID.')
    parser.add_argument('input_file', type=str, help='Path to the input log file')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the function with provided file paths
    parse_llm_output(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
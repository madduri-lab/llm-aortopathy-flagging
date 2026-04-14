# Run the script where file 'PMC_IDs.txt' is, this file should have the PMC IDs you want to download
# Set start_line and end_line to designate which part of the list you want to Run
# If 'SOURCE_DOCUMENTS' directory doesnt exist, the script will create one and output the PDFs there
# A logfile called 'download_logfile.txt' will be generated with status of download for each PDF as successful or failure
# The script should not stall if a download fails, it will continue on to next ID
# This script will print out status of each download incrementally to your terminal
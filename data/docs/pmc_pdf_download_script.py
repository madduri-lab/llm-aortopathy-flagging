from urllib.parse import urlparse  # Import urlparse for handling URLs
from requests_html import HTMLSession  # Import HTMLSession for making HTML requests
import requests
from requests.exceptions import ConnectionError  # Import ConnectionError to handle connection errors
import os  # Import os for file and directory operations
import time  # Import time for timing and delays
import random  # Import random for generating random numbers

s = HTMLSession()  # Create an HTML session

# Set up headers to mimic a browser request
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
}

# Create the 'SOURCE_DOCUMENTS' directory if it doesn't exist
output_dir = 'SOURCE_DOCUMENTS'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Specify the range of lines to read from the input file (1-based indexing)
start_line = 1
end_line = 100

# Open the file containing PMC IDs and read the specified range of lines
with open('PMC_IDs.txt', 'r') as file:
    ids = [line.strip() for line in file.readlines()[start_line - 1:end_line]]

start_time = time.time()  # Start the timer for total download time

request_count = 0  # Initialize a counter for the number of requests

# Open a log file for writing download messages
with open('download_log.txt', 'w') as log:
    # Iterate over each PMC ID
    for pmc in ids:
        download_start_time = time.time()  # Start the timer for individual download
        try:
            pmcid = pmc
            base_url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/'
            # Get the article page
            r = s.get(base_url + pmcid + '/', headers=headers, timeout=5)
            # Find the PDF link on the article page and construct the PDF URL
            pdf_url = 'https://www.ncbi.nlm.nih.gov' + r.html.find('a.int-view', first=True).attrs['href']
            # Download the PDF in chunks and save it to the output directory
            r = s.get(pdf_url, stream=True)
            pdf_filename = os.path.join(output_dir, pmcid + '.pdf')
            with open(pdf_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            download_end_time = time.time()  # Stop the timer for individual download
            download_time = download_end_time - download_start_time  # Calculate individual download time
            # Write success message to log and print it
            log.write(f'Successfully downloaded {pmcid}.pdf in {download_time:.2f} seconds\n')
            print(f'Successfully downloaded {pmcid}.pdf in {download_time:.2f} seconds')

        except ConnectionError as e:
            # Write failure message to log and print it
            log.write(f'Failed to download {pmcid}.pdf\n')
            print(f'Failed to download {pmcid}.pdf')

        time.sleep(5)  # Wait for 5 seconds between each request

        request_count += 1  # Increment the request counter
        if request_count % 15 == 0:  # Check if 15 requests have been made
            random_delay = random.randint(5, 15)  # Generate a random delay between 5 to 15 seconds
            # Print the duration of the random delay
            print(f'Pausing for {random_delay} seconds...')
            time.sleep(random_delay)  # Wait for the randomly generated delay before continuing

end_time = time.time()  # End the timer for total download time
total_time = end_time - start_time
print(f'Total time taken for downloads: {total_time} seconds')  # Print the total time taken for downloads
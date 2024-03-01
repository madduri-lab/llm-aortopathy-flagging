from urllib.parse import urlparse  # Import urlparse for parsing URLs
from requests_html import HTMLSession  # Import HTMLSession for making HTML requests with JavaScript support
import requests  # Import requests for making HTTP requests
from requests.exceptions import ConnectionError  # Import ConnectionError to handle connection errors
import os  # Import os for operating system-dependent functionality, like file and directory operations
import time  # Import time for timing and delays
import random  # Import random for generating random numbers

s = HTMLSession()  # Create an HTML session

# Set up headers to mimic a browser request
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
}

# Create the 'SOURCE_DOCUMENTS' directory if it doesn't exist, this is where all the PDFs will be saved
output_dir = 'SOURCE_DOCUMENTS'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Specify the range of lines to read from the input file (1-based indexing)
start_line = 1
end_line = 5000

# Open the file containing PMC IDs and read the specified range of lines
with open('PMC_IDs.txt', 'r') as file:
    ids = [line.strip() for line in file.readlines()[start_line - 1:end_line]]

start_time = time.time()  # Start the timer for total download time

request_count = 0  # Initialize a counter for the number of requests

# Open a log file for writing download messages
with open('download_logfile.txt', 'w') as log:
    # Iterate over each PMC ID
    for pmc in ids:
        download_start_time = time.time()  # Start the timer for individual download
        try:
            pmcid = pmc
            base_url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/'
            # Get the article page
            r = s.get(base_url + pmcid + '/', headers=headers, timeout=5)
            # Find the PDF link on the article page
            pdf_link = r.html.find('a.int-view', first=True)
            if pdf_link:
                # Construct the PDF URL and download the PDF in chunks
                pdf_url = 'https://www.ncbi.nlm.nih.gov' + pdf_link.attrs['href']
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
            else:
                # Write failure message to log and print it if PDF link is not found
                log.write(f'Failed to find PDF link for {pmcid}\n')
                print(f'Failed to find PDF link for {pmcid}')

        except ConnectionError as e:
            # Write failure message to log and print it if a connection error occurs
            log.write(f'Failed to download {pmcid}.pdf\n')
            print(f'Failed to download {pmcid}.pdf')

        time.sleep(5)  # Wait for 5 seconds between each request

        request_count += 1  # Increment the request counter
        if request_count % 5 == 0:
            # Generate a random delay between 5 to 15 seconds after every 5 requests
            random_delay = random.randint(5, 15)
            print(f'Pausing for {random_delay} seconds...')
            time.sleep(random_delay)  # Wait for the randomly generated delay before continuing

end_time = time.time()  # End the timer for total download time
total_time = end_time - start_time
print(f'Total time taken for downloads: {total_time} seconds')  # Print the total time taken for downloads
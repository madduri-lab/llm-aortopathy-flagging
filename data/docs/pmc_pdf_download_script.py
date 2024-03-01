from urllib.parse import urlparse
from requests_html import HTMLSession
import requests
from requests.exceptions import ConnectionError
import os
import time
import random

s = HTMLSession()

headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
}

output_dir = 'SOURCE_DOCUMENTS'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_line = 58
end_line = 500

with open('PMC_IDs.txt', 'r') as file:
    ids = [line.strip() for line in file.readlines()[start_line - 1:end_line]]

start_time = time.time()

request_count = 0

with open('download_logfile.txt', 'w') as log:
    for pmc in ids:
        download_start_time = time.time()
        try:
            pmcid = pmc
            base_url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/'
            r = s.get(base_url + pmcid + '/', headers=headers, timeout=5)
            pdf_link = r.html.find('a.int-view', first=True)
            if pdf_link:
                pdf_url = 'https://www.ncbi.nlm.nih.gov' + pdf_link.attrs['href']
                r = s.get(pdf_url, stream=True)
                pdf_filename = os.path.join(output_dir, pmcid + '.pdf')
                with open(pdf_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                download_end_time = time.time()
                download_time = download_end_time - download_start_time
                log.write(f'Successfully downloaded {pmcid}.pdf in {download_time:.2f} seconds\n')
                print(f'Successfully downloaded {pmcid}.pdf in {download_time:.2f} seconds')
            else:
                log.write(f'Failed to find PDF link for {pmcid}\n')
                print(f'Failed to find PDF link for {pmcid}')

        except ConnectionError as e:
            log.write(f'Failed to download {pmcid}.pdf\n')
            print(f'Failed to download {pmcid}.pdf')

        time.sleep(5)

        request_count += 1
        if request_count % 5 == 0:
            random_delay = random.randint(5, 15)
            print(f'Pausing for {random_delay} seconds...')
            time.sleep(random_delay)

end_time = time.time()
total_time = end_time - start_time
print(f'Total time taken for downloads: {total_time} seconds')
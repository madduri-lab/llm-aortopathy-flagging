#!/bin/bash

# Specify PDF folder location
# EXAMPLE
# PDF_FOLDER="/home/xx/marfan"
PDF_FOLDER="" 
# Specify output folder location
# EXAMPLE
# OUTPUT_FOLDER="$PDF_FOLDER/xml"
OUTPUT_FOLDER=""

# Check if the xml folder exists, if not create it
if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

# Switch to PDF folder directory
cd "$PDF_FOLDER"

# Iterate through all PDF files in the folder
for pdf in *.pdf; do
    echo "Processing $pdf..."
    # Use the curl command to send a request and save the result to the xml subfolder
    # You may need to change the IP address and port here
    curl -v --form input=@"./$pdf" --form consolidateHeader=1 localhost:8070/api/processFulltextDocument > "$OUTPUT_FOLDER/${pdf%.pdf}.tei.xml"
done

echo "All PDFs processed."

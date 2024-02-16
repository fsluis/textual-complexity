#!/bin/bash

while true; do
    # Run the program and capture the output
    output=$(python pipeline_loop3.py)

    # Check if the output contains "Done"
    if [[ $output == *"Done"* ]]; then
        echo "Program finished successfully."
        break  # Exit the loop if "Done" is found
    else
        echo "Program not yet finished. Running again..."
        sleep 1  # Wait for a second before running the program again
    fi
done

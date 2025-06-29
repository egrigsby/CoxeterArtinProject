#!/bin/bash

job_output=$(sbatch jupyter_job.sl)
echo "$job_output"

job_id=$(echo "$job_output" | awk '{print $4}')
printf "Check job status with:\nsqueue -j ${job_id}\n\n"

# prints all jobs of user
printf "Job Initializing:\n"
squeue -u $USER

sleep 6

# show jobs (now with recently added job)
printf "Updated Job Status:\n"
squeue -u $USER


#jupyter is the name of the slurm job
err_file="logs/jupyter_${job_id}.err"

# Read entire file into a variable
file_contents=$(cat "$err_file")

# Process the file line by line using a while loop and here-string
found_line=false
url=""

while IFS= read -r line; do
    trimmed=$(echo "$line" | sed 's/^[[:space:]]*//')
    if $found_line; then
        url=$(echo "$trimmed" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        break
    fi
    if [[ "$trimmed" == "Or copy and paste one of these URLs:" ]]; then
        found_line=true
    fi
done <<< "$file_contents"

printf "\nCancel job with:\n scancel ${job_id}\n"

# Output URL result
echo ""
if [[ -n "$url" ]]; then
    printf "Jupyter Notebook URL:\n${url}\n\n"
else
    echo "URL not found in $err_file"
fi